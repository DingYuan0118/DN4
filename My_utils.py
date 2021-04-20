def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(q.numel() for q in model.buffers())
    print("\033[1;32;m{}\033[0m model have \033[1;32;m{}\033[0m parameters.".format(model.__class__.__name__, total_params + total_buffers))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\033[1;32;m{}\033[0m model have \033[1;32;m{}\033[0m training parameters.".format(model.__class__.__name__, total_trainable_params))

