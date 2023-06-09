import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from core.metrics import accuracy
from core.utils import SmoothCrossEntropyLoss
from core.utils import track_bn_stats

class HookTool:
    def __init__(self):
        self.fea = None

    def hook_fun(self, module, fea_in, fea_out):
        '''
        注意用于处理feature的hook函数必须包含三个参数[module, fea_in, fea_out]，参数的名字可以自己起，但其意义是
        固定的，第一个参数表示torch里的一个子module，比如Linear,Conv2d等，第二个参数是该module的输入，其类型是
        tuple；第三个参数是该module的输出，其类型是tensor。注意输入和输出的类型是不一样的，切记。
        '''
        self.fea = fea_out

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def _kl_div(logit1, logit2):
    return F.kl_div(F.log_softmax(logit1, dim=1), F.softmax(logit2, dim=1), reduction='batchmean')


def _jensen_shannon_div(logit1, logit2, T=1.):
    prob1 = F.softmax(logit1/T, dim=1)
    prob2 = F.softmax(logit2/T, dim=1)
    mean_prob = 0.5 * (prob1 + prob2)

    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
    jsd = F.kl_div(logsoftmax, prob1, reduction='batchmean')
    jsd += F.kl_div(logsoftmax, prob2, reduction='batchmean')
    return jsd * 0.5


def trades_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, 
                attack='linf-pgd', label_smoothing=0.1, use_cutmix=False, use_consistency=False, cons_lambda=0.0, cons_tem=0.0):
    """
    TRADES training (Zhang et al, 2019).
    """
  
    criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)
    
    x_adv = x_natural.detach() +  torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if use_cutmix: # CutMix
        p_natural = y
    else:
        p_natural = F.softmax(model(x_natural), dim=1)
        p_natural = p_natural.detach()
    
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    elif attack == 'l2-pgd':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()
    track_bn_stats(model, True)
  
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    optimizer.zero_grad()
    # calculate robust loss
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)

    if use_cutmix: # CutMix
        loss_natural = criterion_kl(F.log_softmax(logits_natural, dim=1), y)
    else:
        loss_natural = criterion_ce(logits_natural, y)

    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
    loss = loss_natural + beta * loss_robust

    if use_consistency:
        logits_adv1, logits_adv2 = logits_adv.chunk(2)
        loss = loss + cons_lambda * _jensen_shannon_div(logits_adv1, logits_adv2, cons_tem)
    
    if use_cutmix: # CutMix
        batch_metrics = {'loss': loss.item(), 'clean_acc': -1, 
                     'adversarial_acc': -1}
    else:
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()), 
                     'adversarial_acc': accuracy(y, logits_adv.detach())}
        
    return loss, batch_metrics


def trades_loss_KLFA(model, x_natural, y, optimizer, hook_name, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                attack='linf-pgd', label_smoothing=0.1, use_cutmix=False, use_consistency=False, cons_lambda=0.0,
                cons_tem=0.0):
    """
    TRADES training (Zhang et al, 2019).
    """

    criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)

    def extractor(x, hook_name):
        if hook_name:
            feature = []
            b, _, _, _ = x.shape
            for n, m in model.named_modules():
                if n in hook_name:
                    cur_hook = HookTool()
                    m.register_forward_hook(cur_hook.hook_fun)
                    feature.append(cur_hook)

            _ = model(x)
            feature_vector = torch.zeros(size=(b, 0)).cuda()

            for fea in feature:
                if fea.fea.ndim == 4:
                    tmp = fea.fea.mean(dim=(2, 3))
                feature_vector = torch.cat([feature_vector, tmp], dim=1)

            return feature_vector
        else:
            return model(x)


    x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if use_cutmix:  # CutMix
        p_natural = y
    else:
        p_natural = F.softmax(model(x_natural), dim=1)
        p_natural = p_natural.detach()

    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif attack == 'l2-pgd':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()
    track_bn_stats(model, True)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()
    # calculate robust loss
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)

    if use_cutmix:  # CutMix
        loss_natural = criterion_kl(F.log_softmax(logits_natural, dim=1), y)
    else:
        loss_natural = criterion_ce(logits_natural, y)

    loss_robust = criterion_kl(extractor(x_adv, hook_name), extractor(x_natural, hook_name))
    loss = loss_natural + beta * loss_robust

    if use_consistency:
        logits_adv1, logits_adv2 = logits_adv.chunk(2)
        loss = loss + cons_lambda * _jensen_shannon_div(logits_adv1, logits_adv2, cons_tem)

    if use_cutmix:  # CutMix
        batch_metrics = {'loss': loss.item(), 'clean_acc': -1,
                         "loss_natural": loss_natural.mean().cpu().detach().numpy(),
                         "loss_robust": loss_robust.mean().cpu().detach().numpy(), 'adversarial_acc': -1}
    else:
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()),
                         "loss_natural": loss_natural.mean().cpu().detach().numpy(),
                         "loss_robust": loss_robust.mean().cpu().detach().numpy(),
                         'adversarial_acc': accuracy(y, logits_adv.detach())}

    return loss, batch_metrics


def trades_loss_LSE(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                attack='linf-pgd', label_smoothing=0.1, clip_value=0, use_cutmix=False, num_classes=10):
    """
    SCORE training (Ours).
    """
    # criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    # criterion_kl = nn.KLDivLoss(reduction='sum')
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)
    
    x_adv = x_natural.detach() +  torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if use_cutmix: # CutMix
        p_natural = y
    else:
        p_natural = F.softmax(model(x_natural), dim=1)
        p_natural = p_natural.detach()
    
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            output_adv = F.softmax(model(x_adv), dim=1)
            with torch.enable_grad():
                loss_lse = torch.sum((output_adv - p_natural) ** 2)
            grad = torch.autograd.grad(loss_lse, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif attack == 'l2-pgd':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                output_adv = F.softmax(model(adv), dim=1)
                loss = (-1) * torch.sum((output_adv - p_natural) ** 2)
            loss.backward()

            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()
    track_bn_stats(model, True)
  
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    optimizer.zero_grad()

    if use_cutmix: # CutMix
        y_onehot = y
    else:
        y_onehot = (1 - num_classes * label_smoothing / (num_classes-1)) * F.one_hot(y, num_classes=num_classes) + label_smoothing / (num_classes-1)
    
    logits_natural = F.softmax(model(x_natural), dim=1)
    logits_adv = F.softmax(model(x_adv), dim=1)
    loss_natural = torch.sum((logits_natural - y_onehot) ** 2, dim=-1)
    loss_robust = torch.sum((logits_adv - logits_natural) ** 2, dim=-1)

    loss_robust = F.relu(loss_robust - clip_value) # clip loss value

    loss = loss_natural.mean() + beta * loss_robust.mean()

    if use_cutmix: # CutMix
        batch_metrics = {'loss': loss.item(), 'clean_acc': -1, 
                     'adversarial_acc': -1}
    else:
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()), 
                     'adversarial_acc': accuracy(y, logits_adv.detach())}
        
    return loss, batch_metrics


def trades_loss_LSEM(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                    attack='linf-pgd', label_smoothing=0.1, clip_value=0, use_cutmix=False, num_classes=10):
    """
    SCORE training (Ours).
    """
    # criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    # criterion_kl = nn.KLDivLoss(reduction='sum')
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)

    x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if use_cutmix:  # CutMix
        p_natural = y
    else:
        p_natural = F.softmax(model(x_natural), dim=1)
        p_natural = p_natural.detach()

    loss_momentum = 0

    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            output_adv = F.softmax(model(x_adv), dim=1)
            with torch.enable_grad():
                loss_lse = torch.sum((output_adv - p_natural) ** 2)

            grad = torch.autograd.grad(loss_lse, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            loss_momentum += loss_lse/torch.norm(x_adv - x_natural, p=2)
    elif attack == 'l2-pgd':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta
            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                output_adv = F.softmax(model(adv), dim=1)
                loss = (-1) * torch.sum((output_adv - p_natural) ** 2)
                loss_momentum -= loss
            loss.backward()

            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
            loss_momentum += loss / torch.norm(delta,p=2)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()
    track_bn_stats(model, True)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()

    if use_cutmix:  # CutMix
        y_onehot = y
    else:
        y_onehot = (1 - num_classes * label_smoothing / (num_classes - 1)) * F.one_hot(y,
                                                                                       num_classes=num_classes) + label_smoothing / (
                               num_classes - 1)

    logits_natural = F.softmax(model(x_natural), dim=1)
    logits_adv = F.softmax(model(x_adv), dim=1)
    loss_natural = torch.sum((logits_natural - y_onehot) ** 2, dim=-1)
    loss_robust = torch.sum((logits_adv - logits_natural) ** 2, dim=-1)

    loss_robust = F.relu(loss_robust - clip_value)  # clip loss value

    loss = loss_natural.mean() + beta * loss_robust.mean() + loss_momentum

    if use_cutmix:  # CutMix
        batch_metrics = {'loss': loss.item(), 'clean_acc': -1,
                         'adversarial_acc': -1}
    else:
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()),
                         'adversarial_acc': accuracy(y, logits_adv.detach())}

    return loss, batch_metrics


def trades_loss_LSFA(model, x_natural, y, optimizer, hook_name, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                    attack='linf-pgd', label_smoothing=0.1, clip_value=0, use_cutmix=False, num_classes=10):
    """
    SCORE training (Ours).
    """
    # criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    # criterion_kl = nn.KLDivLoss(reduction='sum')
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)

    def extractor(x, hook_name):
        if hook_name:
            feature = []
            b, _, _, _ = x.shape
            for n, m in model.named_modules():
                if n in hook_name:
                    cur_hook = HookTool()
                    m.register_forward_hook(cur_hook.hook_fun)
                    feature.append(cur_hook)

            _ = model(x)
            feature_vector = torch.zeros(size=(b, 0)).cuda()

            for fea in feature:
                if fea.fea.ndim == 4:
                    tmp = fea.fea.mean(dim=(2, 3))
                feature_vector = torch.cat([feature_vector, tmp], dim=1)

            return feature_vector
        else:
            return model(x)

    x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if use_cutmix:  # CutMix
        p_natural = y
    else:
        p_natural = F.softmax(model(x_natural), dim=1)
        p_natural = p_natural.detach()

    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            output_adv = F.softmax(model(x_adv), dim=1)
            with torch.enable_grad():
                loss_lse = torch.sum((output_adv - p_natural) ** 2)
            grad = torch.autograd.grad(loss_lse, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif attack == 'l2-pgd':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                output_adv = F.softmax(model(adv), dim=1)
                loss = (-1) * torch.sum((output_adv - p_natural) ** 2)
            loss.backward()

            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()
    track_bn_stats(model, True)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()

    if use_cutmix:  # CutMix
        y_onehot = y
    else:
        y_onehot = (1 - num_classes * label_smoothing / (num_classes - 1)) * F.one_hot(y,
                                                                                       num_classes=num_classes) + label_smoothing / (
                               num_classes - 1)

    logits_natural = F.softmax(model(x_natural), dim=1)
    logits_adv = F.softmax(model(x_adv), dim=1)
    loss_natural = torch.sum((logits_natural - y_onehot) ** 2, dim=-1)
    loss_robust = torch.sum((extractor(x_adv,hook_name) - extractor(x_natural,hook_name)) ** 2, dim=-1)

    loss_robust = F.relu(loss_robust - clip_value)  # clip loss value

    loss = loss_natural.mean() + beta * loss_robust.mean()

    if use_cutmix:  # CutMix
        batch_metrics = {'loss': loss.item(), 'clean_acc': -1, "loss_natural": loss_natural.mean().cpu().detach().numpy(),
                         "loss_robust": loss_robust.mean().cpu().detach().numpy(), 'adversarial_acc': -1}
    else:
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()),
                         "loss_natural": loss_natural.mean().cpu().detach().numpy(),"loss_robust": loss_robust.mean().cpu().detach().numpy(),
                         'adversarial_acc': accuracy(y, logits_adv.detach())}

    return loss, batch_metrics


def martfa_loss(model, x_natural, y, optimizer, hook_name, step_size=0.007, epsilon=0.031, perturb_steps=10, beta=6.0,
              attack='linf-pgd'):
    """
    MART training (Wang et al, 2020).
    """

    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)

    def extractor(x, hook_name):
        if hook_name:
            feature = []
            b, _, _, _ = x.shape
            for n, m in model.named_modules():
                if n in hook_name:
                    cur_hook = HookTool()
                    m.register_forward_hook(cur_hook.hook_fun)
                    feature.append(cur_hook)

            _ = model(x)
            feature_vector = torch.zeros(size=(b, 0)).cuda()

            for fea in feature:
                if fea.fea.ndim == 4:
                    tmp = fea.fea.mean(dim=(2, 3))
                feature_vector = torch.cat([feature_vector, tmp], dim=1)

            return feature_vector
        else:
            return model(x)

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise ValueError(f'Attack={attack} not supported for MART training!')
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)
    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(extractor(x_adv, hook_name) + 1e-12), extractor(x_adv, hook_name)), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust

    batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits.detach()),"loss_adv": loss_adv.item(),
                     "loss_robust": loss_robust.item(),'adversarial_acc': accuracy(y, logits_adv.detach())}

    return loss, batch_metrics


