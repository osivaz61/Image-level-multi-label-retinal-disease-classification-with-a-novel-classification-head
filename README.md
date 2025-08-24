Image-level multi-label OIA-ODIR classification (Note: This code implementation is image-level.) 

This model includes the swinV2-Base384, SCA classification head, SNDL loss, and ASAM optimizer. 

**YOU CAN EASILY ADAPT THE SCA CLASSIFICATION HEAD TO YOUR PROBLEMS TO INCREASE THE CLASSIFICATION PERFORMANCE (NOTE: FOR BETTER PERFORMANCE, THE INPUT IMAGE SIZE SHOULD BE 384 OR BIGGER).**

This is the startup model of the proposed model. You need to integrate the warm-up, EMA, and mix-up to achieve better results.

Finally, the dataloader should be revised to return at least four multi-label images per batch, thereby increasing the classification performance with SNDL loss.

needs 4 V100 GPUs (16 GB GPU memory per GPU)

MLDecoder and Shunted Transformer inspire the novel shunted cross-attention (SCA) classification head.

Day by day, it will be updated.
