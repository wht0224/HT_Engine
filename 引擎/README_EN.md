# Low-End GPU Optimized Rendering Engine

![GitHub release](https://img.shields.io/github/release/yourusername/lowend-rendering-engine.svg)
![Python version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Grant reality, unto all beings!

## Project Overview

This rendering engine is designed specifically for low-end GPUs like NVIDIA GTX 750Ti and AMD RX 580. I built it to bring modern rendering capabilities to older hardware, using a range of optimization techniques to achieve visual quality close to high-end GPUs while running on limited resources.

I focused heavily on optimizing for 2GB VRAM limitations, ensuring the engine runs smoothly even on budget graphics cards.

## Core Features

### Hardware Optimization
- Adaptive hardware detection that automatically identifies GPU architectures like GTX 750Ti and RX 580, applying tailored optimizations
- Smart VRAM management with extreme optimizations for 2GB memory, including texture compression and dynamic loading
- Architecture-specific optimizations for Maxwell and GCN instruction sets
- Performance budget system that strictly controls costs at each rendering stage

### Visual Effects System
- Fast approximate global illumination with GPU acceleration, taking less than 1ms per frame
- Optimized screen space reflections with low-resolution, efficient sampling
- Lightweight ambient occlusion that maintains visual quality while reducing performance impact
- Effect manager that intelligently combines and prioritizes effects to achieve the best visual results within budget

### Performance Management
- Multi-level LOD system that automatically generates models with up to 80% polygon reduction for distant objects
- Instanced rendering support for efficient rendering of large numbers of identical geometries
- Automatic batching to reduce state changes and draw calls
- Dynamic quality adjustment that automatically tweaks visual settings based on actual frame rate

### Material System
- Physically-based materials that maintain realism under limited lighting conditions
- Support for high-performance texture compression formats like BC7 and ETC2
- Texture atlasing to reduce texture switching and memory fragmentation

## Supported Hardware

### Primary Target Platforms
- **NVIDIA GTX 750Ti**: 512 CUDA cores, 2GB GDDR5, 86GB/s bandwidth
- **AMD RX 580**: 2304 stream processors, 4GB GDDR5, 256GB/s bandwidth

### Minimum Requirements
- GPU: DX11 or OpenGL 4.5 compatible graphics card with at least 1GB VRAM
- CPU: Dual-core processor
- Memory: 4GB system memory
- OS: Windows 7/10/11, Linux (Ubuntu 18.04+)

## Project Structure

```
Engine/
├── Engine/               # Core engine code
│   ├── Renderer/         # Rendering system
│   │   ├── Effects/      # Optimized visual effects
│   │   ├── Shaders/      # Low-end GPU optimized shaders
│   │   ├── Renderer.py   # Main renderer class
│   │   └── ...
│   ├── Scene/            # Scene management system
│   ├── Resources/        # Resource management system
│   ├── Platform/         # Platform abstraction layer
│   └── Utils/            # Utility classes
├── main.py               # Main entry point
├── start_engine.py       # Engine startup script
├── requirements.txt      # Dependency list
└── setup.py              # Installation script
```

## Installation Guide

### System Requirements

- Python 3.8 or higher
- Ensure graphics card drivers are updated to the latest version

### Dependency Installation

```bash
# Clone repository
# git clone <repository-url>
cd Engine

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies

For best performance, you can install the following optional dependencies:

```bash
# For JIT optimization
pip install numba cython

# For texture compression
pip install moderngl

# For Blender integration (optional)
pip install bpy
```

## Quick Start

### Running the Engine

```bash
# Start the engine
python start_engine.py
```

### Creating a Basic Application

```python
from Engine.Renderer.Renderer import Renderer, RenderQuality
from Engine.Platform.WindowPlatform import WindowPlatform
from Engine.Resources.ResourceManager import ResourceManager
from Engine.Scene.Scene import Scene
from Engine.Scene.Camera import Camera

# Initialize platform
platform = WindowPlatform("Low-End GPU Optimization Demo", 1280, 720)
platform.initialize()

# Initialize resource manager and renderer
resource_manager = ResourceManager(platform)
resource_manager.initialize()

renderer = Renderer(platform, resource_manager)
renderer.initialize()

# Auto-optimize for current hardware
renderer.optimize_for_hardware()

# Create scene and camera
scene = Scene()
camera = Camera()
camera.set_position(0, 2, -5)
camera.look_at(0, 0, 0)
scene.set_active_camera(camera)

# Main loop
while running:
    # Render scene
    renderer.render(scene)
    platform.swap_buffers()
```

## Rendering Optimization Techniques

### 1. Memory Optimization Strategies

- **Texture Compression**: Reduces texture memory usage by 60-80% using BC7/ETC2 formats
- **Texture Streaming**: Dynamically loads/unloads textures based on camera distance
- **Geometry Simplification**: Automatically generates multi-level LODs, using low-poly models for distant objects
- **Instanced Rendering**: Renders 100 identical objects with just 1 draw call

### 2. Shader Optimization

- **Half-Precision Calculations**: Uses FP16 for non-critical calculations to reduce bandwidth usage
- **Instruction Merging**: Optimizes shader instructions to reduce instruction count
- **Precomputed Lighting**: Bakes complex lighting to reduce real-time calculations
- **Selective Effects**: Dynamically enables/disables visual effects based on performance budget

### 3. Low-End GPU Specific Optimizations

#### NVIDIA Maxwell Architecture (GTX 750Ti)
- Optimized texture sampler configuration
- Reduced shader complexity, prioritizing registers over textures
- Bandwidth optimizations for 128-bit memory bus
- Maximum of 4 lights for real-time rendering

#### AMD GCN Architecture (RX 580)
- Optimized compute shaders to fully utilize 2304 stream processors
- Uses asynchronous compute for parallel effect processing
- Texture filtering optimization for 256-bit memory bus
- Supports more lights and higher resolution shadows

## Effect System Guide

### Ambient Occlusion (AO)

```python
# Enable/disable AO
renderer.enable_feature('ambient_occlusion', True)

# Performance impact at different quality levels:
# ULTRA_LOW: ~0.3ms, 64x64 downsampling
# LOW: ~0.5ms, 128x128 downsampling
# MEDIUM: ~0.8ms, 256x256 downsampling
# HIGH: ~1.2ms, 512x512 downsampling
```

### Fast Approximate Global Illumination (FAGI)

```python
# Enable/disable FAGI
renderer.enable_feature('fagi', True)

# Performance impact at different quality levels:
# ULTRA_LOW: ~0.4ms, 128x128 resolution
# LOW: ~0.8ms, 256x256 resolution
# MEDIUM: ~1.5ms, 512x512 resolution
# HIGH: ~2.5ms, 1024x1024 resolution
```

### Screen Space Reflections (SSR)

```python
# Enable/disable SSR
renderer.enable_feature('ssr', True)

# Performance optimizations for low-end GPUs:
# - Reduced sampling count using ray stepping and binary search
# - Lower reflection resolution
# - Limited reflection distance
# - Disabled by default on GTX 750Ti, can be enabled manually
```

## Performance Tuning Guide

### Monitoring Performance Metrics

The renderer collects the following performance metrics:

- **FPS**: Frames per second
- **VRAM Usage**: Video memory consumption (MB)
- **Draw Calls**: Number of draw requests from CPU to GPU
- **Triangle Count**: Number of triangles rendered

### Recommended Settings for GTX 750Ti

```python
# Optimize for GTX 750Ti
renderer.set_render_quality(RenderQuality.LOW)
renderer.set_max_draw_calls(200)
renderer.set_max_visible_lights(4)
renderer.set_shadow_map_resolution(1024)

# Selectively enable features
renderer.enable_feature('ambient_occlusion', True)
renderer.enable_feature('fagi', True)
renderer.enable_feature('ssr', False)  # High impact on GTX 750Ti
```

### Recommended Settings for RX 580

```python
# Optimize for RX 580
renderer.set_render_quality(RenderQuality.MEDIUM)
renderer.set_max_draw_calls(500)
renderer.set_max_visible_lights(8)
renderer.set_shadow_map_resolution(2048)

# Can enable all features
renderer.enable_feature('ambient_occlusion', True)
renderer.enable_feature('fagi', True)
renderer.enable_feature('ssr', True)
```

## Development Guide

### Contributing Code

We welcome Pull Requests to improve the engine! Before submitting, please ensure:

1. Code follows project coding standards
2. Performance has been tested on GTX 750Ti and RX 580
3. Appropriate documentation is provided

## License

This project is licensed under the MIT License - see LICENSE file for details

## Contact

- Email: 13910593150@139.com
---

Grant reality, unto all beings!
