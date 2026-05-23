import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function

# model = models.resnet18()
# inputs = torch.randn(5, 3, 224, 224)

# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     with record_function("model_interface"):
#         model(inputs)

# print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_memeory_usage", row_limit=10))

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

activities=[ProfilerActivity.CPU]

sort_by_keyword = "self_" + torch.device + "_time_total"

with profile(
    activities=activities,
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
) as prof:
    model(inputs)

# Print aggregated stats
print(prof.key_averages(group_by_stack_n=5).table(sort_by=sort_by_keyword, row_limit=2))

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))