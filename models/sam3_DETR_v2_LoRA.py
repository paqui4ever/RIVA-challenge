
import torch
import torch.nn as nn
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print("Warning: 'peft' library not found. LoRA features will not work.")
    LoraConfig = None
    get_peft_model = None

from .sam3_DETR_v2 import Sam3ForClosedSetDetection

class Sam3DETRv2LoRA(Sam3ForClosedSetDetection):
    def __init__(
        self,
        sam3_checkpoint: str = "facebook/sam3",
        num_classes: int = 8,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        # Initialize base model. We pass freeze_sam3=False because we want PEFT to handle 
        # the freezing of the base weights while keeping adapters trainable.
        super().__init__(sam3_checkpoint, num_classes, freeze_sam3=False)

        if LoraConfig is None:
            raise ImportError("Please install 'peft' to use LoRA: pip install peft")

        # Configure LoRA
        # Targeting standard attention projection layers in Transformer-based models (SAM, ViT, etc.)
        peft_config = LoraConfig(
            task_type=None, 
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"], 
            bias="none",
        )

        # Wrap the SAM3 model with LoRA
        self.sam3 = get_peft_model(self.sam3, peft_config)
        
        # Ensure the class head is trainable (it is outside the wrapped self.sam3)
        for param in self.class_embed.parameters():
            param.requires_grad = True

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        # Print PEFT stats for the wrapped backbone
        self.sam3.print_trainable_parameters()
        
        # Calculate global stats including the head
        all_params = 0
        trainable_params = 0
        for _, p in self.named_parameters():
            all_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        
        print(f"Overall Model: Trainable params: {trainable_params} || All params: {all_params} || Trainable%: {100 * trainable_params / all_params:.4f}")
