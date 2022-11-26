### Program implementing the GLIDE Model for text to image generation. This implementation uses classifier-free guidance (CLIP guidance left for another implementation).

## Features:
# 1. This implementation of GLIDE with classifier-free guidance has two main compnents: the text encoder and the diffusion model.
# 2. The text encoder is a transformer encoder with following details:
# 2.1. The input preprocessing involves 3 steps:
# 2.1.1. Creating a sentencepiece based tokenizer from the captions dataset (using COCO captions validation set for this implementation)
# 2.1.2. Tokenizing the captions (appended with <s>, </s> and <pad> tokens)
# 2.1.3. Converting the tokens to embeddings (using nn.Embeddings + positional_embeddings)
# 2.2. After preprocessing, the text embeddings are feeded to the (causally masked) transformer encoder. This involves the following steps:
# 2.2.1. A causal mask (subsequent_mask) and a padding mask is added to the text embeddings
# 2.2.2. The text embeddings and the mask if feeded to the transformer encoder. Let transformer output be 'enc_out' (shape: [batch_size, seq_len, d_model])
# 2.2.3. In addition to enc_out, we also extract the output at index corresponding to <eos> token (this is TODO, currently we just extract output at last [-1] index). Let this be represented by the variable 'eos_out'.
# 3. The diffusion model is based on improved_ddpm paper, with the following details:
# 3.1. Learnable mean and variance {though we predict noise and variance and then calcualate mean from predicted noise}
# 3.2. Hybrid loss = L_simple (mse) + lambda * L_vlb
# 3.3. We use linear schedule for betas (cosine schedule doesn't work for us - TODO: get cosine schedule working)
# 3.4. Strided sampling with CFG based interpolation
# 4. UNet is modified for caption conditioning (following are the modifications:)
# 4.1. The 'eos_out' is added to the time embedding feeded to UNet (this is similar to class_label conditioning scheme)
# 4.2. The 'enc_out' is projected to the dimension of keys / values in the UNet's self_attn layers and concatenated to the keys and values (TODO: take care of dimensions mismatch)

## Todos / Questions:
# 1. Swapping the layernorm order in SublayerConnection of the transformer encoder (this will require a final layernorm in enc_out)
# 2. 'eos_out': should this be last index of 'enc_out' or the index corresponding to the <eos> token
# 3. UNet architecture modifications for caption conditioning (particularly for projecting 'enc_out' and concatenating to keys and values of self_attn layers in UNet)
# 4. Get the cosine schedule working
# 5. Add checkpointing to save and load intermediate model_dict for training
# 6. Do we need a learning schedule with warmup_steps
# 7. Add a testing script to test the final (trained) model by generating images for different input prompts / captions.
# 8. Separate implementation using CLIP guidance instead of classifier-free guidance
# 9. img preprocessing: does it make a difference if img.shape: [c,h,w] versus [c,w,h]
# 10. img preprocessing: should we normalize by mean,std=((.5,.5,.5), (.5,.5,.5)) or using imagenet mean,std=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# 11. img preprocessing: remember to 'unnormalize' the image when viewing at the end (as a post-processing step after sampling from model - else we get saturated looking imgs)
# 12. Ways to reduce GPU memory footprint of the program


import os
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import unidecode
import sentencepiece as spm

from text_encoder_utils import *
from unet_utils import *


def linear_noise_schedule(beta_min, beta_max, max_time_steps):
    return torch.linspace(beta_min, beta_max, max_time_steps)

# the function used to calculate cosine factor used in cosine noise schedule
def cosine_func(t, max_time_steps, s=0.008):
    return torch.pow( torch.cos( (((t/max_time_steps)+s) / (1+s)) * torch.tensor(torch.pi/2) ), 2)

def cosine_noise_schedule(max_time_steps):
    betas = []
    # initial values
    f_0 = cosine_func(0, max_time_steps)
    alpha_hat_prev = 1.
    # iterate
    for t in range(1, max_time_steps+1):
        f_t = cosine_func(t, max_time_steps)
        alpha_hat = f_t / f_0
        beta = 1 - (alpha_hat/alpha_hat_prev)
        beta = torch.clamp(beta, min=0., max=0.999)
        betas.append(beta)
        alpha_hat_prev = alpha_hat
    return torch.stack(betas, dim=0)

# OpenAi implementation for cosine schedule
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    betas = np.array(betas)
    betas = torch.from_numpy(betas).float()
    return betas

def noise_img(img_ori, alphas_hat, t, device):
    sqrt_alpha_hat = torch.sqrt(alphas_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alphas_hat[t])[:, None, None, None]
    eps = torch.randn_like(img_ori)
    noised_img = ( sqrt_alpha_hat * img_ori ) + ( sqrt_one_minus_alpha_hat * eps )
    return noised_img, eps

# function to calculate KL div between two gaussians - TODO: check dims for diagonal variance
# used to calculate L_t = KL(q_posterior, p)
def kl_normal(mean_q, logvar_q, mean_p, logvar_p):
    # stop gradient on means
    mean_q, mean_p = mean_q.detach(), mean_p.detach()
    return 0.5 * ( -1.0 + logvar_p - logvar_q + torch.exp(logvar_q - logvar_p) + ((mean_q - mean_p)**2) * torch.exp(-logvar_p) )

### functions to calculate L_0 = -log p(x_0 | x_1) - borrowed from OpenAi implementation of improved DDPM

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

# utility function to take mean of a tensor across all dimensions except the first (batch) dimension
def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.shape))))

# function to calculate mean and variance of q_posterior: q(x_t-1 | x_t,x_0)
def q_posterior_mean_variance(x_0, x_t, t, t_minus1, alphas_hat, alphas, betas):
    alpha_hat = alphas_hat[t][:, None, None, None]
    alpha_hat_minus1 = alphas_hat[t_minus1][:, None, None, None]
    # its necessary to re-calculate beta and alpha from alphas_hat:
    alpha = alpha_hat / alpha_hat_minus1
    beta = 1 - alpha
    mean = ( torch.sqrt(alpha_hat_minus1) * beta * x_0 + torch.sqrt(alpha) * (1 - alpha_hat_minus1) * x_t ) / (1 - alpha_hat)
    var = ( (1 - alpha_hat_minus1) * beta ) / (1 - alpha_hat)
    logvar = torch.log(var)
    return mean, logvar

# function to calculate mean and variance of p(x_t-1 | x_t)
def p_mean_variance(net, x_t, t, t_minus1, eos_out, enc_out, alphas_hat):
    out = net(x_t, t, eos_out, enc_out) # the unet predicts the concatenated [mean, frac]
    img_channels = x_t.shape[1]
    pred_noise, frac = torch.split(out, img_channels, dim=1)
    # shift frac values to be in [0, 1] from [-1, 1]
    frac = (frac + 1) / 2.0
    # calculate log variance using frac interpolatiion between min_log_var (beta_tilde) and max_log_var (beta)
    alpha_hat = alphas_hat[t][:, None, None, None]
    alpha_hat_minus1 = alphas_hat[t_minus1][:, None, None, None]
    # so its necessary to re-calculate beta from alphas_hat:
    beta = 1 - (alpha_hat / alpha_hat_minus1)
    alpha = 1 - beta
    beta_tilde = ( (1 - alpha_hat_minus1) * beta ) / (1 - alpha_hat)
    max_logvar = torch.log(beta)
    min_logvar = torch.log(beta_tilde)
    logvar = frac * max_logvar + (1 - frac) * min_logvar
    # calculate mean from pred_noise
    mean = ( x_t - ((beta * pred_noise) / torch.sqrt(1 - alpha_hat)) ) / torch.sqrt(alpha)
    return mean, logvar, pred_noise

# function to calculate hybrid loss: L_hybrid = L_simple + lambda * L_vlb
def calculate_hybrid_loss(net, x_0, t, eos_out, enc_out, L_lambda, alphas_hat, alphas, betas, device):
    x_t, true_noise = noise_img(x_0, alphas_hat, t, device)
    q_mean, q_logvar = q_posterior_mean_variance(x_0, x_t, t, t-1, alphas_hat, alphas, betas)
    p_mean, p_logvar, pred_noise = p_mean_variance(net, x_t, t, t-1, eos_out, enc_out, alphas_hat)
    # for t == 1:
    p_log_scale = 0.5 * p_logvar
    L_vlb_0 = -1 * discretized_gaussian_log_likelihood(x_0, p_mean, p_log_scale)
    L_vlb_0 = L_vlb_0 / torch.log(torch.tensor(2.0)) # convert loss from nats to bits
    L_vlb_0 = mean_flat(L_vlb_0) # take mean across all dims except batch_dim # shape: [batch_size]
    L_simple_0 = torch.pow(pred_noise - true_noise, 2)
    L_simple_0 = mean_flat(L_simple_0)
    L_hybrid_0 = L_vlb_0
    # L_hybrid_0 = L_simple_0

    # for t > 1:
    L_simple = torch.pow(pred_noise - true_noise, 2) # mse loss but don't want to reduce mean or sum
    L_simple = mean_flat(L_simple) # take mean across all dims except batch_dim # shape: [batch_size]
    L_vlb_t = kl_normal(q_mean, q_logvar, p_mean, p_logvar)
    L_vlb_t = L_vlb_t / torch.log(torch.tensor(2.0)) # convert loss from nats to bits
    L_vlb_t = mean_flat(L_vlb_t) # take mean across all dims except batch_dim # shape: [batch_size]
    L_hybrid_t = L_simple + L_lambda * L_vlb_t
    # populate final loss vector according to t values
    L_hybrid = torch.where((t == 1), L_hybrid_0, L_hybrid_t) # shape: [batch_size]
    L_hybrid = L_hybrid.mean() # final loss scalar
    return L_hybrid

# function to sample x_t-1 ~ p(x_t-1 | x_t)
def p_sample_CFG(i, net, x_t, t, t_minus1, eos_label, enc_label, alphas_hat, guidance_strength):
    mean_cond, logvar_cond, pred_noise_cond = p_mean_variance(net, x_t, t, t_minus1, eos_label, enc_label, alphas_hat)
    mean_uncond, logvar_uncond, pred_noise_uncond = p_mean_variance(net, x_t, t, t_minus1, None, None, alphas_hat)
    # calculated interpolated mean (weighted by guidance strength)
    mean_interpolated = mean_cond + guidance_strength * ( mean_cond - mean_uncond )
    # sample
    eps = torch.randn_like(x_t)
    if i == 1:
        eps = eps * 0
    x_t_minus1 = mean_interpolated + torch.exp(0.5 * logvar_cond) * eps
    return x_t_minus1

# strided sampling (with classifier free guidance based interpolation)
def sample_strided_CFG(net, alphas_hat, guidance_strength, max_time_steps, subseq_steps, img_size, device, n, eos_label, enc_label):
    net.eval()
    with torch.no_grad():
        subseq = torch.linspace(0, max_time_steps-1, subseq_steps, dtype=torch.int)
        x = torch.randn((n, 3, img_size, img_size)).to(device)
        for i in tqdm( reversed(range(1, subseq_steps)), position=0 ):
            tau = (torch.ones(n) * subseq[i]).long().to(device)
            tau_minus1 = (torch.ones(n) * subseq[i-1]).long().to(device)
            x = p_sample_CFG(i, net, x, tau, tau_minus1, eos_label, enc_label, alphas_hat, guidance_strength)
    net.train()
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x

# note that we sample n=batch_size images for a same label (class label doesn't change between the n samples)
# so we use class label as subfolder name
def save_img_CFG(img, name):
    name = 'generated_v1/' + name
    grid = torchvision.utils.make_grid(img)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(name)

# # fetch dataset - using data loader
# def get_data(img_size, datapath, batch_size):
#     transforms = torchvision.transforms.Compose([
#         torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
#         torchvision.transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # equivalent to transforming pixel values from range [0,1] to [-1,1]
#     ])
#     dataset = torchvision.datasets.ImageFolder(datapath, transform=transforms)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     return dataloader

# utility function to load img and captions data
def load_data():
    imgs_folder = 'dataset_coco_val2017/images/'
    captions_file_path = 'dataset_coco_val2017/annotations/captions_val2017.json'
    captions_file = open(captions_file_path)
    captions = json.load(captions_file)
    img_dict, img_cap_dict = {}, {}
    max_caption_len = 0
    for img in captions['images']:
        id, file_name = img['id'], img['file_name']
        img_dict[id] = file_name
    for cap in captions['annotations']:
        id, caption = cap['image_id'], cap['caption']
        # update max_caption_len
        caption_len = len(caption.split(' '))
        if caption_len > max_caption_len:
            max_caption_len = caption_len
        # process caption
        caption = unidecode.unidecode(caption) # strip accents
        caption = caption.lower()
        # use img_name as key for img_cap_dict
        img = img_dict[id]
        img_cap_dict[img] = caption
    max_caption_len += 3 # for <s>, </s> and a precautionary <pad>
    return img_cap_dict, max_caption_len

# utility function to preprocess minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions
def preprocess_minibatch(minibatch, spm_processor, sos_token, eos_token, pad_token, max_seq_len, img_size, device):
    augmented_imgs, tokenized_captions = [], []
    imgs_folder = 'dataset_coco_val2017/images/'
    for img_filename, caption_text in minibatch:
        # tokenize caption text
        caption_tokens = spm_processor.encode(caption_text, out_type=int)
        caption_tokens = [sos_token] + caption_tokens + [eos_token] # append sos and eos tokens
        while len(caption_tokens) < max_seq_len:
            caption_tokens.append(pad_token) # padding
        tokenized_captions.append(caption_tokens)
        # obtain augmented img from img_filename
        img_path = imgs_folder + img_filename
        img = cv2.imread(img_path, 1)
        resize_shape = (img_size, img_size)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) / 255
        img = torch.tensor(img)
        img = img.transpose(1,2).transpose(0,1).transpose(1,2) # [w,h,c] -> [c,h,w]
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize( int(1.25*img_size) ),  # image_size + 1/4 * image_size
            torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0)),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
        ])
        img = transforms(img)
        augmented_imgs.append(img)
    augmented_imgs = torch.stack(augmented_imgs, dim=0).to(device)
    tokenized_captions = torch.LongTensor(tokenized_captions).to(device)
    return augmented_imgs, tokenized_captions


# utility function to save a checkpoint (model_state and optimizer_state) - saves on whatever device the model was training on
def save_checkpoint(checkpoint_path, text_encoder, net, optimizer):
    save_dict = {'text_encoder_state_dict': text_encoder.state_dict(),
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    torch.save(save_dict, checkpoint_path)

# utility function to load checkpoint to resume training from - loads to the device passed as 'device' argument
def load_checkpoint(checkpoint_path, text_encoder, net, optimizer, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    text_encoder.load_state_dict(ckpt['text_encoder_state_dict'])
    net.load_state_dict(ckpt['net_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    text_encoder.train()
    net.train()
    return text_encoder, net, optimizer

### main
if __name__ == '__main__':

    # hperparams for text_encoder (transformer)
    d_model = 512
    d_k = 64
    d_v = 64
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    dropout = 0.1

    # hyperparams for diffusion model
    L_lambda = 0.001 # for weighing L_vlb
    guidance_strength = 3 # w in classifier free guidance paper
    p_uncond = 0.2 # probability for setting class_label = None
    beta_min = 1e-4 # not needed for cosine noise schedule
    beta_max = 0.02 # not needed for cosine noise schedule
    max_time_steps = 1000 # 4000
    subseq_steps = 200 # used for both strided sampling and ddim sampling - should be less than max_time_steps
    img_size = 64
    lr = 3e-4
    batch_size = 2
    max_epochs = 30000 * 10
    random_seed = 10

    checkpoint_path = 'ckpts_v1/latest.pt' # path to a saved checkpoint (model state and optimizer state) to resume training from
    resume_training_from_ckpt = True

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # spm processor as tokenizer / detokenizer
    spm_processor = spm.SentencePieceProcessor(model_file='spm1.model')
    vocab_size = len(spm_processor)
    # declare sos, eos, unk and pad tokens
    sos_token, eos_token, unk_token = spm_processor.piece_to_id(['<s>', '</s>', '<unk>'])
    pad_token = unk_token # <unk> token is used as the <pad> token

    # load data (img_filenames and caption_text) and create img_cap_dict
    img_cap_dict, max_seq_len = load_data()

    # init text encoder (transformer)
    text_encoder = init_text_encoder(vocab_size, max_seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, pad_token, device).to(device)

    # init UNet (with modifications for GLIDE with classifier-free guidance)
    net = UNet_Glide(c_in=3, c_out=6, caption_emb_dim=d_model, device=device).to(device)

    # calcualate betas and alphas
    betas = linear_noise_schedule(beta_min, beta_max, max_time_steps)
    # betas = cosine_noise_schedule(max_time_steps)
    # betas = betas_for_alpha_bar(max_time_steps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    alphas = 1 - betas
    alphas_hat = torch.cumprod(alphas, dim=0)
    alphas_hat = alphas_hat.to(device)
    betas = betas.to(device)
    alphas = alphas.to(device)

    # optimizer and loss criterion
    opt_params = list(net.parameters()) + list(text_encoder.parameters())
    optimizer = torch.optim.AdamW(params=opt_params, lr=lr)

    # load checkpoint to resume training from
    if resume_training_from_ckpt:
        text_encoder, net, optimizer = load_checkpoint(checkpoint_path, text_encoder, net, optimizer, device)

    # train
    for ep in tqdm(range(max_epochs)):

        # fetch minibatch - a batch of [img_filename, caption_text] pairs
        minibatch_keys = np.random.choice(list(img_cap_dict.keys()), size=batch_size)
        minibatch = [[k, img_cap_dict[k]] for k in minibatch_keys]

        # preprocess minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions
        imgs, captions = preprocess_minibatch(minibatch, spm_processor, sos_token, eos_token, pad_token, max_seq_len, img_size, device) # imgs.shape:[batch_size, 3, 64, 64], captions.shape:[batch_size, max_seq_len]

        # encode captions using the text_encoder to obtain enc_out and eos_out vectors
        enc_out, eos_out = text_encoder(captions) # enc_out.shape:[batch_size, max_seq_len, d_model], eos_out.shape:[batch_size, d_model]

        # set caption embeddings = None with prob p_uncond
        eos_label = eos_out[0].unsqueeze(0) # used while sampling
        enc_label = enc_out[0].unsqueeze(0) # used while sampling
        if np.random.rand() < p_uncond:
            eos_out = None
            enc_out = None

        t = torch.randint(low=1, high=max_time_steps, size=(batch_size,)).to(device) # sample a time step uniformly in [1, max_time_steps)

        loss = calculate_hybrid_loss(net, imgs, t, eos_out, enc_out, L_lambda, alphas_hat, alphas, betas, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % (max_epochs//200) == 0:
            print('ep:{} \t loss:{}'.format(ep, loss.item()))

            # save checkpoint
            save_checkpoint(checkpoint_path, text_encoder, net, optimizer)

            ## sample
            text_encoder.eval()
            net.eval()

            sample_caption_text = minibatch[0][1]
            sampled_img = sample_strided_CFG(net, alphas_hat, guidance_strength, max_time_steps, subseq_steps, imgs.shape[-1], device, 1, eos_label, enc_label)
            # save sampled_img
            save_img_CFG(sampled_img, str(ep) + ': ' + sample_caption_text + '.png')

            text_encoder.train()
            net.train()
