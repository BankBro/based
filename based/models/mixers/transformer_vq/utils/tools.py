import torch
import torch.nn.functional as F

def one_hot_encode(x:torch.tensor, num_classes, dtype, device):
    *main_shape, len_shape = x.shape

    one_hot = torch.zeros(
        *main_shape, len_shape, num_classes,
        dtype=dtype, device=device)  # (*main_shape, len_shape, num_classes)

    valid_mask = (x >= 0) & (x < num_classes)  # (*main_shape, len_shape)
    one_hot[valid_mask] = F.one_hot(x[valid_mask], num_classes=num_classes)
    
    return one_hot

def check_dtypes_equal(*tensors):
    first_dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        assert tensor.dtype == first_dtype, (
            f"Tensor dtype mismatch: {tensor.dtype} vs {first_dtype}"
        )

def check_tensor_shape(tensor, shape):
    assert tensor.shape == shape, (
        f"Tensor shape mismatch: got: {tensor.shape}, expected: {shape}."
    )