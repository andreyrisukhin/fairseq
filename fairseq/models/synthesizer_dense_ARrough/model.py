from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    register_model,
    register_model_architecture,
)
import fairseq.models.transformer as transformer
# from fairseq.models.abc.abc_encoder import AbcEncoder
# from fairseq.models.abc.abc_decoder import AbcDecoder
# from fairseq.models.abc.utils import add_abc_args, get_abc_attr
from fairseq.modules.multihead_synthesizer import *

@register_model("synthesizer_dense")
class SynthesizerDenseModel(transformer.TransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        super(AbcModel, AbcModel).add_args(parser)
        # add_abc_args(parser) # TODO Why was a custom parser needed for ABC? To handle additional arguments?

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = transformer.DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = transformer.DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", transformer.DEFAULT_MIN_PARAMS_TO_WRAP
            )
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)
        cls.tie_parameters(
            args,
            encoder,
            decoder,
            share_cross_causal=args.phi_share_cross_causal,
            share_first_layer=args.phi_share_first_layer,
            share_last_layer=args.phi_share_last_layer
        )
        return cls(args, encoder, decoder)

    
    # @classmethod
    # def build_encoder(cls, args, src_dict, embed_tokens):
    #     return AbcEncoder(
    #         args, 
    #         src_dict, 
    #         embed_tokens, 
    #         has_cross_attention=True)
    
    """
    This file (in abc) =/= mha.py
    3 file hierarchy
    reuse likely most here; synth and template also replace attn with our version
    can likely reuse the model by renaming "abcencoder->synthencoder"

    diff arguments for attn, may need arg update in old repo (=motivation for new pieces in ABC)

    in abc encode/decode can reuse; should modify the file focusing on attn
    Find this; contains new implementation of softmax, maybe in seperate file
        focus on this! Reuse others
    """

    # @classmethod
    # def build_decoder(cls, args, tgt_dict, embed_tokens):
    #     return AbcDecoder(
    #         args,
    #         tgt_dict,
    #         embed_tokens
    #     )

    # @classmethod
    # def tie_parameters(
    #     cls,
    #     args,
    #     encoder,
    #     decoder,
    #     share_cross_causal: bool = True,
    #     share_first_layer: bool = True,
    #     share_last_layer: bool = False
    # ):
    #     if args.phi_func == "mlp":
    #         if share_first_layer:
    #             proj1 = (encoder
    #                     .cross_attention_project
    #                     .layers[0]
    #                     .phi_func.mlp.proj1)
    #             # layernorm = (encoder
    #             #              .cross_attention_project
    #             #              .layers[0]
    #             #              .phi_func.mlp.layernorm)
    #             for layer in (encoder
    #                         .cross_attention_project
    #                         .layers[1:]):
    #                 layer.phi_func.mlp.proj1 = proj1
    #                 # layer.phi_func.mlp.layernorm = layernorm
    #             if share_cross_causal:
    #                 assert args.causal_proj_dim == args.cross_proj_dim
    #             else:
    #                 proj1 = (decoder.layers[0]
    #                         .self_attn.phi_func.mlp.proj1)
    #                 # layernorm = (decoder.layers[0]
    #                 #              .self_attn.phi_func.mlp.layernorm)
    #             for layer in (decoder.layers):
    #                 layer.self_attn.phi_func.mlp.proj1 = proj1
    #                 # layer.self_attn.phi_func.mlp.layernorm = layernorm
    #         if share_last_layer:
    #             proj2 = (encoder
    #                     .cross_attention_project
    #                     .layers[0]
    #                     .phi_func.mlp.proj2)
    #             scale = (encoder
    #                     .cross_attention_project
    #                     .layers[0]
    #                     .phi_func.mlp.scale)
    #             for layer in (encoder
    #                         .cross_attention_project
    #                         .layers[1:]):
    #                 layer.phi_func.mlp.proj2 = proj2
    #                 layer.phi_func.mlp.scale = scale
    #             if share_cross_causal:
    #                 assert args.causal_proj_dim == args.cross_proj_dim
    #             else:
    #                 proj2 = (decoder.layers[0]
    #                         .self_attn.phi_func.mlp.proj2)
    #                 scale = (decoder.layers[0]
    #                         .self_attn.phi_func.mlp.scale)
    #             for layer in (decoder.layers):
    #                 layer.self_attn.phi_func.mlp.proj2 = proj2
    #                 layer.self_attn.phi_func.mlp.scale = scale
    #     elif args.phi_func == "linformer":
    #         linformer = (encoder.cross_attention_project.layers[0].phi_func.linformer)
    #         for layer in (encoder.cross_attention_project.layers[1:]):
    #             layer.phi_func.linformer = linformer
    #         if share_cross_causal:
    #             assert args.causal_proj_dim == args.cross_proj_dim
    #         else:
    #             linformer = (decoder.layers[0].self_attn.phi_func.linformer)
    #         for layer in (decoder.layers):
    #             layer.self_attn.phi_func.linformer = linformer
            
                
    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        prev_output_tokens,
        features_only: bool = False,
        **unused
    ):
        """
        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only
        )
        return decoder_out


@register_model_architecture("abc", "abc")
def base_architecture(args):
    transformer.base_architecture(args)
    get_abc_attr(args)
    

@register_model_architecture("abc", "abc_iwslt_de_en")
def transformer_iwslt_de_en(args):
    transformer.transformer_iwslt_de_en(args)
    base_architecture(args)
    

@register_model_architecture("abc", "abc_wmt_en_de")
def transformer_wmt_en_de(args):
    base_architecture(args)