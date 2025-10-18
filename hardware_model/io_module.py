class IOModule:
    def __init__(self, bandwidth, latency):
        self.bandwidth = bandwidth
        self.latency = latency
    def info(self):
        return f"bandwidth: {self.bandwidth/1000/1000/1000}GB/s, latency: {self.latency*1e6}us"

IO_module_dict = {
    "A100": IOModule(2039e9, 1e-6),
    "TPUv3": IOModule(float("inf"), 1e-6),
    "MI210": IOModule(1.6e12, 1e-6),
    "PIMSAB": IOModule(2039e9, 1e-6),
    "SIMDRAM": IOModule(2039e9, 1e-6),
}
