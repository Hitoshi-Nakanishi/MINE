def loss(y_pred, y_true, mu, logvar, beta):
    BCE = F.cross_entropy(y_pred, y_true)
    KLD = beta * 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def layer_KLD(layer):
    layer_w,