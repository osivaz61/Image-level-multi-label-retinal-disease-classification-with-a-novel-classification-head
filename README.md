Image-level multi-label OIA-ODIR classification

This model includes the swinV2-Base384, SCA classification head, SNDL loss, and ASAM optimizer. 

This is the startup model of the proposed model. You need to integrate the warm-up, EMA, and mix-up to achieve better results.

Finally, the dataloader should be revised to return at least four multi-label images per batch, thereby increasing the classification performance with SNDL loss.

needs 4 V100 GPUs (16 GB GPU memory per GPU)

Note: This code implementation is image-level.

The novel shunted cross-attention (SCA) classification head is inspired by MLDecoder and Shunted Transformer.

Day by day, it will be updated.
