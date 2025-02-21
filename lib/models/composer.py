import torch

def batch_heatmap_generator_vis(
    joints : torch.Tensor, 
    joints_vis : torch.Tensor, 
    heatmap_size : int, 
    image_size : int, 
    sigma : int,
):
    """
    joints :     (batch n_view n_joints 2)
    joints_vis : (batch n_view n_joints)
    """
    batch, n_view, n_joints = joints.shape[:3]
    device = joints.device
    
    heatmap = torch.zeros(batch * n_view * n_joints, heatmap_size, heatmap_size).float().to(device)

    tmp = sigma * 3

    stride = image_size / heatmap_size
    mu_x = (joints[:, :, :, 0] / stride + 0.5).to(torch.long)
    mu_y = (joints[:, :, :, 1] / stride + 0.5).to(torch.long)
    ux, ly = mu_x - tmp, mu_y - tmp
    bx, ry = mu_x + tmp + 1, mu_y + tmp + 1
    ex_idx = torch.where((ux>=heatmap_size) | (ly>=heatmap_size) | (bx<0) | (ry<0))
    if len(ex_idx[0])>0:
        joints_vis[ex_idx] = 0

    heat_size = 2 * tmp + 1
    X = torch.arange(0, heat_size, 1).float()
    Y = X[:, None]

    gauss = torch.exp(-((X-tmp)**2 + (Y-tmp)**2) / (2 * sigma**2))

    g_x_l, g_x_r = torch.where(ux>0, 0, -ux), torch.where(bx>heatmap_size, heatmap_size, bx) - ux
    g_y_l, g_y_r = torch.where(ly>0, 0, -ly), torch.where(ry>heatmap_size, heatmap_size, ry) - ly
    im_x_l, im_x_r = torch.where(ux>0, ux, 0), torch.where(bx>heatmap_size, heatmap_size, bx)
    im_y_l, im_y_r = torch.where(ly>0, ly, 0), torch.where(ry>heatmap_size, heatmap_size, ry)

    joints_vis = joints_vis.reshape(-1)
    im_x_l = im_x_l.reshape(-1)
    im_x_r = im_x_r.reshape(-1)
    im_y_l = im_y_l.reshape(-1)
    im_y_r = im_y_r.reshape(-1)
    g_x_l = g_x_l.reshape(-1)
    g_x_r = g_x_r.reshape(-1)
    g_y_l = g_y_l.reshape(-1)
    g_y_r = g_y_r.reshape(-1)

    joints_mask = joints_vis <= 0.5
    im_x_l[joints_mask] = 0
    im_x_l[joints_mask] = 0
    im_y_l[joints_mask] = 0
    im_y_r[joints_mask] = 0
    g_x_l[joints_mask] = 0
    g_x_r[joints_mask] = 0
    g_y_l[joints_mask] = 0
    g_y_r[joints_mask] = 0

    batch_product, H, W = heatmap.shape
    im_x_range = torch.arange(W).view(1, 1, -1).expand(batch_product, H, W).to(device)
    im_y_range = torch.arange(H).view(1, -1, 1).expand(batch_product, H, W).to(device)

    mask_im_x = (im_x_range >= im_x_l.view(-1, 1, 1)) & (im_x_range < im_x_r.view(-1, 1, 1))
    mask_im_y = (im_y_range >= im_y_l.view(-1, 1, 1)) & (im_y_range < im_y_r.view(-1, 1, 1))
    mask_im = mask_im_x & mask_im_y

    GH, GW = gauss.shape
    gauss = gauss[None].expand(batch_product, GH, GW).to(device)
    g_x_range = torch.arange(GW).view(1, 1, -1).expand(batch_product, GH, GW).to(device)
    g_y_range = torch.arange(GH).view(1, -1, 1).expand(batch_product, GH, GW).to(device)

    mask_g_x = (g_x_range >= g_x_l.view(-1, 1, 1)) & (g_x_range < g_x_r.view(-1, 1, 1))
    mask_g_y = (g_y_range >= g_y_l.view(-1, 1, 1)) & (g_y_range < g_y_r.view(-1, 1, 1))
    mask_g = mask_g_x & mask_g_y
    heatmap[mask_im] = gauss[mask_g]
    
    return heatmap.reshape(batch, n_view, n_joints, heatmap_size, heatmap_size), joints_vis.reshape(batch, n_view, n_joints)

def batch_heatmap_generator(
    joints : torch.Tensor, 
    joints_vis : torch.Tensor, 
    heatmap_size : int, 
    image_size : int, 
    sigma : int,
):
    """
    joints :     (batch n_view n_joints 2)
    joints_vis : (batch n_view n_joints)
    """
    batch, n_view, n_joints = joints.shape[:3]
    device = joints.device
    
    heatmap = torch.zeros(batch * n_view * n_joints, heatmap_size, heatmap_size).float().to(device)

    tmp = sigma * 3

    stride = image_size / heatmap_size
    mu_x = (joints[:, :, :, 0] / stride + 0.5).to(torch.long)
    mu_y = (joints[:, :, :, 1] / stride + 0.5).to(torch.long)
    ux, ly = mu_x - tmp, mu_y - tmp
    bx, ry = mu_x + tmp + 1, mu_y + tmp + 1
    ex_idx = torch.where((ux>=heatmap_size) | (ly>=heatmap_size) | (bx<0) | (ry<0))
    if len(ex_idx[0])>0:
        joints_vis[ex_idx] = 0

    heat_size = 2 * tmp + 1
    X = torch.arange(0, heat_size, 1).float()
    Y = X[:, None]

    gauss = torch.exp(-((X-tmp)**2 + (Y-tmp)**2) / (2 * sigma**2))

    g_x_l, g_x_r = torch.where(ux>0, 0, -ux), torch.where(bx>heatmap_size, heatmap_size, bx) - ux
    g_y_l, g_y_r = torch.where(ly>0, 0, -ly), torch.where(ry>heatmap_size, heatmap_size, ry) - ly
    im_x_l, im_x_r = torch.where(ux>0, ux, 0), torch.where(bx>heatmap_size, heatmap_size, bx)
    im_y_l, im_y_r = torch.where(ly>0, ly, 0), torch.where(ry>heatmap_size, heatmap_size, ry)

    joints_vis = joints_vis.reshape(-1)
    im_x_l = im_x_l.reshape(-1)
    im_x_r = im_x_r.reshape(-1)
    im_y_l = im_y_l.reshape(-1)
    im_y_r = im_y_r.reshape(-1)
    g_x_l = g_x_l.reshape(-1)
    g_x_r = g_x_r.reshape(-1)
    g_y_l = g_y_l.reshape(-1)
    g_y_r = g_y_r.reshape(-1)

    joints_mask = joints_vis <= 0.5
    im_x_l[joints_mask] = 0
    im_x_l[joints_mask] = 0
    im_y_l[joints_mask] = 0
    im_y_r[joints_mask] = 0
    g_x_l[joints_mask] = 0
    g_x_r[joints_mask] = 0
    g_y_l[joints_mask] = 0
    g_y_r[joints_mask] = 0

    batch_product, H, W = heatmap.shape
    im_x_range = torch.arange(W).view(1, 1, -1).expand(batch_product, H, W).to(device)
    im_y_range = torch.arange(H).view(1, -1, 1).expand(batch_product, H, W).to(device)

    mask_im_x = (im_x_range >= im_x_l.view(-1, 1, 1)) & (im_x_range < im_x_r.view(-1, 1, 1))
    mask_im_y = (im_y_range >= im_y_l.view(-1, 1, 1)) & (im_y_range < im_y_r.view(-1, 1, 1))
    mask_im = mask_im_x & mask_im_y

    GH, GW = gauss.shape
    gauss = gauss[None].expand(batch_product, GH, GW).to(device)
    g_x_range = torch.arange(GW).view(1, 1, -1).expand(batch_product, GH, GW).to(device)
    g_y_range = torch.arange(GH).view(1, -1, 1).expand(batch_product, GH, GW).to(device)

    mask_g_x = (g_x_range >= g_x_l.view(-1, 1, 1)) & (g_x_range < g_x_r.view(-1, 1, 1))
    mask_g_y = (g_y_range >= g_y_l.view(-1, 1, 1)) & (g_y_range < g_y_r.view(-1, 1, 1))
    mask_g = mask_g_x & mask_g_y
    heatmap[mask_im] = gauss[mask_g]
    
    return heatmap.reshape(batch, n_view, n_joints, heatmap_size, heatmap_size)

def heatmap_generator(
    joints : torch.Tensor, 
    joints_vis : torch.Tensor, 
    heatmap_size : int, 
    image_size : int, 
    sigma : int
):
    """
    joints :     (n_joints 2)
    joints_vis : (n_joints)
    """
    n_joints = joints.shape[0]
    device = joints.device
    
    heatmap = torch.zeros(n_joints, heatmap_size, heatmap_size).float().to(device)

    tmp = sigma * 3

    stride = image_size / heatmap_size
    mu_x = (joints[:, 0] / stride + 0.5).to(torch.long)
    mu_y = (joints[:, 1] / stride + 0.5).to(torch.long)
    ux, ly = mu_x - tmp, mu_y - tmp
    bx, ry = mu_x + tmp + 1, mu_y + tmp + 1
    ex_idx = torch.where((ux>=heatmap_size) | (ly>=heatmap_size) | (bx<0) | (ry<0))
    if len(ex_idx[0])>0:
        joints_vis[ex_idx] = 0

    heat_size = 2 * tmp + 1
    X = torch.arange(0, heat_size, 1).float()
    Y = X[:, None]

    gauss = torch.exp(-((X-tmp)**2 + (Y-tmp)**2) / (2 * sigma**2))

    g_x_l, g_x_r = torch.where(ux>0, 0, -ux), torch.where(bx>heatmap_size, heatmap_size, bx) - ux
    g_y_l, g_y_r = torch.where(ly>0, 0, -ly), torch.where(ry>heatmap_size, heatmap_size, ry) - ly
    im_x_l, im_x_r = torch.where(ux>0, ux, 0), torch.where(bx>heatmap_size, heatmap_size, bx)
    im_y_l, im_y_r = torch.where(ly>0, ly, 0), torch.where(ry>heatmap_size, heatmap_size, ry)

    # TODO 高速化
    for i in range(n_joints):
        if joints_vis[i]>0.5:
            ixl, ixr = im_x_l[i], im_x_r[i]
            iyl, iyr = im_y_l[i], im_y_r[i]
            gxl, gxr = g_x_l[i], g_x_r[i] 
            gyl, gyr = g_y_l[i], g_y_r[i] 
            heatmap[i, iyl:iyr, ixl:ixr] = gauss[gyl:gyr, gxl:gxr]
    
    return heatmap, joints_vis

def affine_transform(x, trans):
    return torch.einsum('bvij,bvkj->bvki', trans, x)