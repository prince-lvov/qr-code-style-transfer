from typing import List, Union

import brevitas.quant
import torch
from torch import nn
from torch import Tensor


import brevitas
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int32Bias
from brevitas.quant_tensor import QuantTensor


class QuantAE(nn.Module):
  
    def __init__(self,
        name: str,
        in_channels: int,
        latent_dim: int,
        image_size: int,
        encoder_dims: List = None,
        decoder_dims: List = None,
        encoder_out_channel_sizes: List = None,              
        act_quant: brevitas.quant = Int8ActPerTensorFloat,
        weight_quant: brevitas.quant = Int8WeightPerTensorFloat,
        return_quant_tensor: bool = False, #True,
        debug: bool = False,
        **kwargs
    ) -> None:                
        super().__init__()

        self.name = name
        self.latent_dim = latent_dim

        self.image_size = image_size
        self.encoder_dims = encoder_dims

        self.encoder_out_channel_size = encoder_out_channel_sizes[self.encoder_dims[-1]]      
        self.decoder_dims = decoder_dims

        self.debug = debug

        self.encoder = _get_encoder(
            in_channels, self.encoder_dims,
            act_quant=act_quant, weight_quant=weight_quant, return_quant_tensor=return_quant_tensor)

        self.hidden, self.decoder_input, self.decoder_input_batchnorm = _get_hidden_layer_and_decode_input(
            self.encoder_dims, self.encoder_out_channel_size, self.latent_dim, self.decoder_dims,
            act_quant=act_quant, weight_quant=weight_quant, return_quant_tensor=return_quant_tensor)

        self.decoder = _get_decoder(
            self.decoder_dims,
            act_quant=act_quant, weight_quant=weight_quant, return_quant_tensor=return_quant_tensor)

        # self.flatten_identity1 and self.flatten_identity2 are used to encapsulate the `torch.flatten`.
        self.flatten_identity1 = qnn.QuantIdentity(act_quant=act_quant, return_quant_tensor=return_quant_tensor)
        self.flatten_identity2 = qnn.QuantIdentity(act_quant=act_quant, return_quant_tensor=return_quant_tensor)

  
    def forward(self, input: Union[Tensor, QuantTensor], **kwargs) -> Union[Tensor, QuantTensor]:
        result = self.encoder(input)
        if self.debug: 
            print("size after encoder", result.shape)

        # As `torch.flatten` is a PyTorch layer, you must place it between two `QuantIdentity`
        # layers to ensure that all intermediate values of the network are properly quantized.
        result = self.flatten_identity1(result)
        result = torch.flatten(result, start_dim=1)
        result = self.flatten_identity2(result)
        if self.debug: 
            print("size after flatten", result.shape)
        
        result = self.hidden(result)      
        if self.debug: 
            print("size after hidden", result.shape)

        result = self.decoder_input(result)
        result = self.decoder_input_batchnorm(result)
        if self.debug: 
            print("size after decoder_input", result.shape)

        result = self.decoder(result)
        if self.debug: 
            print("size after decoder", result.shape)

        # Convert flat tensor to 2D
        result = result.view(-1, 3, self.image_size, self.image_size)
        if self.debug: 
            print("size after Convert flat tensor to 2D (result.view)", result.shape)

        return result #result.value if isinstance(result, QuantTensor) else result


def _get_encoder(
        in_channels: int,
        encoder_dims: List,
        act_quant: brevitas.quant,
        weight_quant: brevitas.quant,
        return_quant_tensor: bool) -> nn.Sequential:
    modules = []

    for h_dim in encoder_dims:
        modules.append(
          qnn.QuantIdentity(act_quant=act_quant, return_quant_tensor=return_quant_tensor)
        )

        modules.append(
          # Quant analog of: nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
          qnn.QuantConv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1,
                          #weight_bit_width=8,
                          weight_quant=weight_quant,
                          #output_quant=act_quant,
                          return_quant_tensor=return_quant_tensor)
        )
        
        modules.append(
        # #   # Quant analog of: nn.BatchNorm2d(h_dim)
          nn.BatchNorm2d(h_dim)
        )
        
        modules.append(
        #   # No quant analog of: nn.LeakyReLU()
          nn.LeakyReLU()
        )

        in_channels = h_dim

    return nn.Sequential(*modules)


def _get_decoder(
        decoder_dims: List,
        act_quant: brevitas.quant,
        weight_quant: brevitas.quant,
        return_quant_tensor: bool) -> nn.Sequential:
    modules = []

    qlinear_args={
        "weight_bit_width": 8,
        "weight_quant": weight_quant,
        "input_quant": act_quant,
        "bias": True, # - increase FHE compile complexity
        "bias_quant": Int32Bias,
        "narrow_range": True,
        "return_quant_tensor": return_quant_tensor
    }

    for i in range(len(decoder_dims) - 1):
        # modules.append(
        #   qnn.QuantIdentity(act_quant=act_quant, return_quant_tensor=return_quant_tensor)
        # )

        # No quant analog of: 
        # nn.ConvTranspose2d(encoder_dims[i], encoder_dims[i + 1], kernel_size=3, stride = 2, padding=1, output_padding=0), # 1 was changed to 0 to get the same 290x290 output
        modules.append(          
          qnn.QuantLinear(
            in_features=decoder_dims[i], 
            out_features=decoder_dims[i+1],
            **qlinear_args)     
        )

        modules.append(
          nn.BatchNorm1d(decoder_dims[i + 1])
        )
        
        modules.append(
          nn.LeakyReLU()
        )       
            
    return nn.Sequential(*modules)


def _get_hidden_layer_and_decode_input(
        encoder_dims: List,
        encoder_out_channel_size: int,
        latent_dim: int,
        decoder_dims: List,
        act_quant: brevitas.quant,
        weight_quant: brevitas.quant,
        return_quant_tensor: bool) -> Union[Tensor, QuantTensor]:
    
    encoder_output_pixel_count = encoder_dims[-1] * encoder_out_channel_size[0] * encoder_out_channel_size[1]
    print(f"Linear parameters: encoder output pixel count: {encoder_output_pixel_count}, latent dim.: {latent_dim}")

    qlinear_args={
        "weight_bit_width": 8,
        "weight_quant": weight_quant,
        "input_quant": act_quant,
        "bias": True, # - increase FHE compile complexity
        "bias_quant": Int32Bias,
        "narrow_range": True,
        "return_quant_tensor": return_quant_tensor
    }
    
    # Quant analog of: self.hidden = nn.Linear(encoder_dims[-1] * self.encoder_out_channel_size[0] * self.encoder_out_channel_size[1], latent_dim)
    hidden = qnn.QuantLinear(
        in_features=encoder_output_pixel_count, 
        out_features=latent_dim,
        **qlinear_args)
    
    # Quant analog of: self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.encoder_out_channel_size[0] * self.encoder_out_channel_size[1])
    decoder_input = qnn.QuantLinear(
        in_features=latent_dim, 
        out_features=decoder_dims[0],
        **qlinear_args)
    
    decoder_input_batchnorm = nn.BatchNorm1d(decoder_dims[0])

    return hidden, decoder_input, decoder_input_batchnorm
