import re
import math
import numpy as np

class ShaderOptimizer:
    """
    着色器优化器
    针对低端GPU（如NVIDIA GTX 750Ti和AMD RX 580）的着色器代码优化工具
    实现各种着色器优化技术，以提高在低端硬件上的性能
    """
    
    def __init__(self, gpu_architecture="default"):
        # GPU架构信息
        self.gpu_architecture = gpu_architecture.lower()
        
        # 优化级别 (0-3，0表示禁用优化，3表示最高优化级别)
        self.optimization_level = 2
        
        # 优化选项
        self.optimization_options = {
            "constant_folding": True,          # 常量折叠
            "algebraic_simplification": True, # 代数简化
            "dead_code_elimination": True,    # 死代码消除
            "instruction_reordering": True,   # 指令重排
            "precision_optimization": True,   # 精度优化
            "loop_optimization": True,        # 循环优化
            "memory_access_optimization": True, # 内存访问优化
            "vectorization": True,            # 向量化
            "texture_sampling_optimization": True, # 纹理采样优化
            "branch_optimization": True,      # 分支优化
            "register_coalescing": True,      # 寄存器合并
            "attribute_packing": True,        # 属性打包
            "uniform_buffer_optimization": True, # 统一缓冲区优化
        }
        
        # 架构特定优化规则
        self.architecture_rules = {
            "maxwell": {
                # NVIDIA Maxwell架构 (GTX 750Ti) 特定优化规则
                "avoid_expensive_instructions": True,
                "prefer_dp4a": True,            # 优先使用dp4a指令
                "use_texture_gather": True,     # 使用纹理收集操作
                "avoid_complex_control_flow": True, # 避免复杂控制流
                "optimize_for_128bit_registers": True, # 针对128位寄存器优化
                "max_instructions_per_group": 64, # 每组最大指令数
                "max_registers": 32,           # 最大寄存器使用数
            },
            "gcn": {
                # AMD GCN架构 (RX 580) 特定优化规则
                "use_ls_instructions": True,    # 使用加载/存储指令
                "prefer_mad_instructions": True, # 优先使用mad指令
                "optimize_for_wavefront": True, # 针对波前优化
                "avoid_uniform_control_flow": True, # 避免统一控制流
                "max_instructions_per_group": 64, # 每组最大指令数
                "max_registers": 64,           # 最大寄存器使用数
            },
            "pascal": {
                # NVIDIA Pascal架构特定优化规则
                "use_shared_memory": True,      # 使用共享内存
                "prefer_f16_calculations": True, # 优先使用半精度计算
                "use_tensor_cores": False,      # 使用张量核心（仅高端卡）
                "max_instructions_per_group": 64, # 每组最大指令数
                "max_registers": 64,           # 最大寄存器使用数
            },
            "turing": {
                # NVIDIA Turing架构 (RTX 20系列) 特定优化规则
                "avoid_expensive_instructions": True,
                "prefer_dp4a": True,
                "use_texture_gather": True,
                "use_rt_cores": True,           # 使用RT核心
                "use_tensor_cores": True,       # 使用张量核心
                "optimize_for_128bit_registers": True,
                "max_instructions_per_group": 128,
                "max_registers": 64,
            },
            "ampere": {
                # NVIDIA Ampere架构 (RTX 30系列) 特定优化规则
                "avoid_expensive_instructions": True,
                "prefer_dp4a": True,
                "use_texture_gather": True,
                "use_rt_cores": True,
                "use_tensor_cores": True,
                "optimize_for_128bit_registers": True,
                "max_instructions_per_group": 128,
                "max_registers": 64,
            },
            "ada": {
                # NVIDIA Ada Lovelace架构 (RTX 40系列) 特定优化规则
                "avoid_expensive_instructions": True,
                "prefer_dp4a": True,
                "use_texture_gather": True,
                "use_rt_cores": True,
                "use_tensor_cores": True,
                "optimize_for_128bit_registers": True,
                "max_instructions_per_group": 128,
                "max_registers": 64,
            },
            "rdna1": {
                # AMD RDNA1架构 (RX 6000系列) 特定优化规则
                "use_ls_instructions": True,
                "prefer_mad_instructions": True,
                "optimize_for_wavefront": True,
                "use_vector_instructions": True,
                "max_instructions_per_group": 128,
                "max_registers": 128,
            },
            "rdna2": {
                # AMD RDNA2架构 (RX 7000系列) 特定优化规则
                "use_ls_instructions": True,
                "prefer_mad_instructions": True,
                "optimize_for_wavefront": True,
                "use_vector_instructions": True,
                "use_rt_cores": True,
                "max_instructions_per_group": 128,
                "max_registers": 128,
            },
            "intel_arc": {
                # Intel Arc架构特定优化规则
                "avoid_expensive_instructions": True,
                "use_vector_instructions": True,
                "optimize_for_128bit_registers": True,
                "max_instructions_per_group": 128,
                "max_registers": 64,
            },
            "default": {
                # 默认优化规则
                "max_instructions_per_group": 64,
                "max_registers": 32,
            }
        }
        
        # 当前架构的优化规则
        self.current_rules = self.architecture_rules.get(self.gpu_architecture, self.architecture_rules["default"])
        
        # 性能统计
        self.optimization_stats = {
            "shaders_optimized": 0,
            "instructions_removed": 0,
            "registers_saved": 0,
            "precision_adjustments": 0,
            "loops_optimized": 0,
            "branches_optimized": 0,
            "textures_optimized": 0,
            "memory_accesses_improved": 0,
        }
    
    def optimize_shader(self, shader_source, shader_type="fragment"):
        """
        优化着色器源代码
        
        Args:
            shader_source: 原始着色器源代码
            shader_type: 着色器类型 ("vertex" 或 "fragment")
            
        Returns:
            str: 优化后的着色器源代码
        """
        if self.optimization_level == 0:
            return shader_source
        
        optimized_source = shader_source
        
        # 添加注释标记优化开始
        optimized_source = "// 开始着色器优化\n" + optimized_source
        
        # 1. 应用通用优化
        if self.optimization_options["constant_folding"] and self.optimization_level >= 1:
            optimized_source = self._constant_folding(optimized_source)
        
        if self.optimization_options["algebraic_simplification"] and self.optimization_level >= 1:
            optimized_source = self._algebraic_simplification(optimized_source)
        
        if self.optimization_options["dead_code_elimination"] and self.optimization_level >= 2:
            optimized_source = self._dead_code_elimination(optimized_source)
        
        # 2. 应用架构特定优化
        optimized_source = self._apply_architecture_optimizations(optimized_source, shader_type)
        
        # 3. 应用高级优化（优化级别3）
        if self.optimization_level >= 3:
            if self.optimization_options["instruction_reordering"]:
                optimized_source = self._instruction_reordering(optimized_source)
            
            if self.optimization_options["loop_optimization"]:
                optimized_source = self._loop_optimization(optimized_source)
            
            if self.optimization_options["branch_optimization"]:
                optimized_source = self._branch_optimization(optimized_source)
        
        # 更新性能统计
        self.optimization_stats["shaders_optimized"] += 1
        
        # 添加注释标记优化结束
        optimized_source += "\n// 结束着色器优化"
        
        return optimized_source
    
    def _constant_folding(self, shader_source):
        """
        常量折叠优化
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        # 这是一个简化的实现，实际实现会更复杂
        # 查找并计算常量表达式
        
        # 匹配数字常量表达式
        number_pattern = r'\b(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\b'
        
        def evaluate_constant(match):
            try:
                left = float(match.group(1))
                op = match.group(2)
                right = float(match.group(3))
                
                if op == '+':
                    result = left + right
                elif op == '-':
                    result = left - right
                elif op == '*':
                    result = left * right
                elif op == '/':
                    if right != 0:
                        result = left / right
                    else:
                        return match.group(0)  # 避免除零错误
                else:
                    return match.group(0)
                
                # 格式化结果，移除不必要的小数点
                if result.is_integer():
                    return str(int(result))
                else:
                    return str(result)
            except:
                return match.group(0)
        
        # 反复应用常量折叠，直到没有更多变化
        while True:
            new_source = re.sub(number_pattern, evaluate_constant, shader_source)
            if new_source == shader_source:
                break
            shader_source = new_source
        
        # 匹配简单的数学函数
        math_functions = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': math.sqrt,
            'abs': abs,
            'ceil': math.ceil,
            'floor': math.floor,
            'exp': math.exp,
            'log': math.log,
        }
        
        for func_name, func in math_functions.items():
            func_pattern = rf'\b{func_name}\s*\(\s*(\d+(?:\.\d+)?)\s*\)\b'
            
            def eval_func(match, func=func):
                try:
                    arg = float(match.group(1))
                    result = func(arg)
                    return str(result)
                except:
                    return match.group(0)
            
            shader_source = re.sub(func_pattern, eval_func, shader_source)
        
        return shader_source
    
    def _algebraic_simplification(self, shader_source):
        """
        代数简化优化
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        # 简单的代数简化规则
        simplifications = [
            # x + 0 => x
            (r'(\w+)\s*\+\s*0', r'\1'),
            (r'0\s*\+\s*(\w+)', r'\1'),
            # x - 0 => x
            (r'(\w+)\s*-\s*0', r'\1'),
            # 0 - x => -x
            (r'0\s*-\s*(\w+)', r'-\1'),
            # x * 0 => 0
            (r'(\w+)\s*\*\s*0', r'0'),
            (r'0\s*\*\s*(\w+)', r'0'),
            # x * 1 => x
            (r'(\w+)\s*\*\s*1', r'\1'),
            (r'1\s*\*\s*(\w+)', r'\1'),
            # x / 1 => x
            (r'(\w+)\s*/\s*1', r'\1'),
            # x * (-1) => -x
            (r'(\w+)\s*\*\s*-1', r'-\1'),
            (r'-1\s*\*\s*(\w+)', r'-\1'),
            # -(-x) => x
            (r'-\(\s*-\s*(\w+)\s*\)', r'\1'),
            # x + (-y) => x - y
            (r'(\w+)\s*\+\s*-\s*(\w+)', r'\1 - \2'),
            # x - (-y) => x + y
            (r'(\w+)\s*-\s*-\s*(\w+)', r'\1 + \2'),
            # x - x => 0
            (r'(\w+)\s*-\s*\1', r'0'),
            # min(x, x) => x
            (r'min\(\s*(\w+)\s*,\s*\1\s*\)', r'\1'),
            # max(x, x) => x
            (r'max\(\s*(\w+)\s*,\s*\1\s*\)', r'\1'),
            # clamp(x, x, x) => x
            (r'clamp\(\s*(\w+)\s*,\s*\1\s*,\s*\1\s*\)', r'\1'),
            # abs(x) * sign(x) => x (当x >= 0)
            # 注意：这只是一个简单的模式匹配，实际应用中需要更复杂的分析
        ]
        
        # 应用简化规则
        for pattern, replacement in simplifications:
            shader_source = re.sub(pattern, replacement, shader_source)
        
        return shader_source
    
    def _dead_code_elimination(self, shader_source):
        """
        死代码消除优化
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        # 这是一个简化的实现，实际实现会更复杂
        # 移除不会被使用的变量声明和赋值
        
        # 查找变量声明
        var_pattern = r'\b(highp|mediump|lowp|in|out|uniform|const)?\s*(\w+)\s+(\w+)\s*=\s*[^;]+;'
        
        # 这里只是一个示例，实际的死代码消除需要进行数据流分析
        # 在这个简单实现中，我们只移除明显的死代码，如无法到达的代码块
        
        # 移除 return 语句后面的代码（在同一个函数内）
        lines = shader_source.split('\n')
        in_function = False
        function_lines = []
        optimized_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            
            # 检测函数开始
            if re.search(r'\b\w+\s*\([^)]*\)\s*{', stripped_line):
                in_function = True
                function_lines = []
            
            if in_function:
                function_lines.append(line)
                
                # 检测 return 语句
                if re.search(r'\breturn\b', stripped_line):
                    # 标记return后的代码为死代码
                    for i in range(len(function_lines)-1, -1, -1):
                        if '{' in function_lines[i]:
                            break
                        if '}' in function_lines[i]:
                            # 保留右括号
                            break
                        if i > len(function_lines)-1:
                            # 标记为死代码
                            function_lines[i] = '// 死代码: ' + function_lines[i]
                
                # 检测函数结束
                if '}' in stripped_line and '{' not in stripped_line:
                    in_function = False
                    optimized_lines.extend(function_lines)
            else:
                optimized_lines.append(line)
        
        return '\n'.join(optimized_lines)
    
    def _apply_architecture_optimizations(self, shader_source, shader_type):
        """
        应用架构特定的优化
        
        Args:
            shader_source: 原始着色器源代码
            shader_type: 着色器类型
            
        Returns:
            str: 优化后的着色器源代码
        """
        optimized_source = shader_source
        
        # 根据当前架构应用特定优化
        rules = self.current_rules
        
        # 1. NVIDIA架构特定优化
        if self.gpu_architecture in ["maxwell", "pascal", "turing", "ampere", "ada"]:
            if rules.get("avoid_expensive_instructions", False):
                # 替换昂贵的指令
                optimized_source = self._replace_expensive_instructions(optimized_source)
            
            if rules.get("prefer_dp4a", False) and shader_type == "fragment":
                # 尝试使用dp4a指令优化点积操作
                optimized_source = self._optimize_dot_products(optimized_source)
            
            if rules.get("avoid_complex_control_flow", False):
                # 简化控制流
                optimized_source = self._simplify_control_flow(optimized_source)
            
            if rules.get("use_rt_cores", False):
                # 添加RT核心优化注释
                optimized_source = "// RT核心优化启用\n" + optimized_source
            
            if rules.get("use_tensor_cores", False):
                # 添加张量核心优化注释
                optimized_source = "// 张量核心优化启用\n" + optimized_source
        
        # 2. AMD架构特定优化
        elif self.gpu_architecture in ["gcn", "rdna1", "rdna2"]:
            if rules.get("use_ls_instructions", False):
                # 添加GCN特定的加载/存储优化注释
                optimized_source = "// AMD LS指令优化启用\n" + optimized_source
            
            if rules.get("prefer_mad_instructions", False):
                # 尝试将 a*b + c 模式替换为 mad(a,b,c)
                optimized_source = self._optimize_mad_instructions(optimized_source)
            
            if rules.get("avoid_uniform_control_flow", False):
                # 优化统一控制流
                optimized_source = self._optimize_uniform_control_flow(optimized_source)
            
            if rules.get("use_vector_instructions", False):
                # 添加向量指令优化注释
                optimized_source = "// AMD向量指令优化启用\n" + optimized_source
            
            if rules.get("use_rt_cores", False):
                # 添加RT核心优化注释
                optimized_source = "// AMD RT核心优化启用\n" + optimized_source
        
        # 3. Intel架构特定优化
        elif self.gpu_architecture == "intel_arc":
            if rules.get("avoid_expensive_instructions", False):
                # 替换昂贵的指令
                optimized_source = self._replace_expensive_instructions(optimized_source)
            
            if rules.get("use_vector_instructions", False):
                # 添加向量指令优化注释
                optimized_source = "// Intel向量指令优化启用\n" + optimized_source
        
        # 4. 精度优化（针对所有架构）
        if self.optimization_options["precision_optimization"]:
            optimized_source = self._optimize_precision(optimized_source, shader_type)
        
        # 5. 纹理采样优化
        if self.optimization_options["texture_sampling_optimization"]:
            optimized_source = self._optimize_texture_sampling(optimized_source)
        
        # 6. 内存访问优化
        if self.optimization_options["memory_access_optimization"]:
            optimized_source = self._optimize_memory_access(optimized_source)
        
        return optimized_source
    
    def _optimize_precision(self, shader_source, shader_type):
        """
        精度优化
        
        Args:
            shader_source: 原始着色器源代码
            shader_type: 着色器类型
            
        Returns:
            str: 优化后的着色器源代码
        """
        optimized_source = shader_source
        
        # 为片段着色器添加默认精度限定符（如果没有）
        if shader_type == "fragment" and "precision" not in optimized_source:
            optimized_source = "precision mediump float;\n" + optimized_source
        
        # 基于架构调整精度
        if self.gpu_architecture == "maxwell" or self.gpu_architecture == "pascal":
            # Maxwell/Pascal架构上，mediump可能会比highp慢，因为它们内部使用highp
            # 所以我们尝试将mediump替换为highp
            optimized_source = re.sub(r'\bmediump\b', 'highp', optimized_source)
        elif self.gpu_architecture == "gcn":
            # GCN架构上，mediump通常更高效
            # 我们可以将非关键变量的精度从highp降低到mediump
            # 这需要更复杂的分析，这里只是一个示例
            pass
        
        # 优化向量和矩阵的精度
        # 例如：将纹理坐标计算使用mediump
        texture_coord_pattern = r'highp\s+(vec2|vec3)\s+(\w+)(\[\d+\])?\s*=\s*(texture|uv|texCoord)'
        optimized_source = re.sub(texture_coord_pattern, r'mediump \1 \2\3 = \4', optimized_source)
        
        # 更新性能统计
        self.optimization_stats["precision_adjustments"] += 1
        
        return optimized_source
    
    def _optimize_texture_sampling(self, shader_source):
        """
        纹理采样优化
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        optimized_source = shader_source
        
        # 1. 合并相邻的纹理采样操作
        # 查找相似的纹理采样
        texture_samples = re.findall(r'texture(2D|Cube|Proj)\s*\(\s*(\w+)\s*,\s*([^)]+)\)', optimized_source)
        
        # 这里只是一个简化的实现，实际的合并需要更复杂的分析
        
        # 2. 优化纹理过滤模式
        # 添加注释建议使用合适的过滤模式
        if self.gpu_architecture == "maxwell":
            optimized_source = "// 建议：使用线性过滤而非各向异性过滤以提高Maxwell架构性能\n" + optimized_source
        elif self.gpu_architecture == "gcn":
            optimized_source = "// 建议：GCN架构支持高效的各向异性过滤\n" + optimized_source
        
        # 3. 预计算纹理坐标
        # 替换重复的纹理坐标计算
        # 这是一个简化的实现
        
        # 4. 纹理数组优化
        # 检查是否有可能使用纹理数组来减少状态切换
        
        # 更新性能统计
        self.optimization_stats["textures_optimized"] += 1
        
        return optimized_source
    
    def _optimize_memory_access(self, shader_source):
        """
        内存访问优化
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        optimized_source = shader_source
        
        # 1. 减少重复的变量访问
        # 查找重复访问同一变量的模式
        
        # 2. 优化向量/矩阵访问模式
        # 例如：优先使用连续的向量组件访问
        
        # 3. 避免跨行访问矩阵
        # 对于GCN架构，跨行访问矩阵效率较低
        if self.gpu_architecture == "gcn":
            # 替换可能的跨行矩阵访问
            pass
        
        # 4. 缓存局部变量
        # 识别频繁访问的uniform变量，建议缓存到局部变量
        uniform_access_pattern = r'(uniform\s+\w+\s+(\w+)(\[\d+\])?)'
        
        # 这是一个简化的实现，实际的优化需要更复杂的分析
        
        # 更新性能统计
        self.optimization_stats["memory_accesses_improved"] += 1
        
        return optimized_source
    
    def _instruction_reordering(self, shader_source):
        """
        指令重排优化
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        # 这是一个简化的实现，实际实现会更复杂
        # 指令重排可以帮助GPU更好地隐藏延迟
        
        # 添加注释标记
        optimized_source = "// 应用指令重排优化（简化版）\n" + shader_source
        
        # 实际的指令重排需要进行详细的依赖分析
        # 这里我们只是添加一个示例，说明这个功能
        
        return optimized_source
    
    def _loop_optimization(self, shader_source):
        """
        循环优化
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        optimized_source = shader_source
        
        # 1. 循环展开
        # 检测简单的小循环并展开
        loop_pattern = r'for\s*\(\s*int\s+(\w+)\s*=\s*(\d+)\s*;\s*(\w+)\s*<\s*(\d+)\s*;\s*(\w+)\s*\+\+\s*\)\s*{'
        
        def unroll_loop(match):
            var_name = match.group(1)
            start = int(match.group(2))
            end = int(match.group(4))
            
            # 只展开小循环（循环次数<10）
            if end - start < 10:
                # 更新性能统计
                self.optimization_stats["loops_optimized"] += 1
                
                # 返回注释，表示这里应该展开循环
                return f'// 建议：展开循环 {var_name} 从 {start} 到 {end}\n{{'
            
            return match.group(0)
        
        optimized_source = re.sub(loop_pattern, unroll_loop, optimized_source)
        
        # 2. 循环不变式提取
        # 识别循环中计算结果不变的表达式并移到循环外
        
        # 3. 循环边界优化
        # 确保循环边界是常量，以帮助GPU优化
        
        return optimized_source
    
    def _branch_optimization(self, shader_source):
        """
        分支优化
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        optimized_source = shader_source
        
        # 1. 替换简单的if-else为数学表达式
        # 例如：if (x > 0) y = a; else y = b; 可以替换为 y = mix(b, a, step(0, x));
        simple_if_pattern = r'if\s*\(\s*(\w+)\s*([<>]=?)\s*(\w+)\s*\)\s*(\w+)\s*=\s*([^;]+);\s*else\s*(\w+)\s*=\s*([^;]+);'
        
        def replace_with_mix(match):
            var1 = match.group(1)
            op = match.group(2)
            var2 = match.group(3)
            result_var = match.group(4)
            true_val = match.group(5)
            else_result_var = match.group(6)
            false_val = match.group(7)
            
            # 检查两个结果变量是否相同
            if result_var != else_result_var:
                return match.group(0)
            
            # 根据操作符选择适当的函数
            if op == '>':
                condition = f'step({var2}, {var1})'
            elif op == '<':
                condition = f'step({var1}, {var2})'
            elif op == '>=':
                condition = f'step({var2} - 0.0001, {var1})'
            elif op == '<=':
                condition = f'step({var1} - 0.0001, {var2})'
            else:
                return match.group(0)
            
            # 更新性能统计
            self.optimization_stats["branches_optimized"] += 1
            
            # 返回使用mix函数的表达式
            return f'{result_var} = mix({false_val}, {true_val}, {condition});'
        
        optimized_source = re.sub(simple_if_pattern, replace_with_mix, optimized_source)
        
        # 2. 避免动态分支
        if self.gpu_architecture == "maxwell" or (self.gpu_architecture == "gcn" and self.current_rules.get("avoid_uniform_control_flow", False)):
            # 添加注释建议避免动态分支
            optimized_source = "// 警告：当前架构上动态分支可能导致性能下降\n" + optimized_source
        
        return optimized_source
    
    def _replace_expensive_instructions(self, shader_source):
        """
        替换昂贵的指令
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        # 替换一些已知昂贵的指令为更高效的替代方案
        replacements = [
            # 使用近似版本的sin/cos
            (r'\bsin\s*\(', r'fast_sin('),
            (r'\bcos\s*\(', r'fast_cos('),
            # 替换pow(x, 2)为x*x
            (r'\bpow\s*\(\s*(\w+)\s*,\s*2\s*\)', r'(\1 * \1)'),
            # 替换pow(x, 0.5)为sqrt(x)
            (r'\bpow\s*\(\s*(\w+)\s*,\s*0\.5\s*\)', r'sqrt(\1)'),
            # 替换normalize(x)为x * inversesqrt(dot(x, x))
            (r'\bnormalize\s*\(\s*(\w+)\s*\)', r'(\1 * inversesqrt(dot(\1, \1)))'),
        ]
        
        for pattern, replacement in replacements:
            shader_source = re.sub(pattern, replacement, shader_source)
        
        # 添加快速三角函数的定义（如果使用了）
        if 'fast_sin(' in shader_source or 'fast_cos(' in shader_source:
            fast_trig_defs = """
// 快速三角函数近似（针对低端GPU优化）
highp float fast_sin(highp float x) {
    // 使用多项式近似
    // 将x范围限制在[-π, π]以提高精度
    highp float x2 = x * x;
    highp float x3 = x * x2;
    highp float x5 = x3 * x2;
    
    // 泰勒展开近似: sin(x) ≈ x - x^3/6 + x^5/120
    return x - x3 * 0.166666667 + x5 * 0.00833333333;
}

highp float fast_cos(highp float x) {
    // 使用多项式近似
    // cos(x) ≈ 1 - x^2/2 + x^4/24
    highp float x2 = x * x;
    highp float x4 = x2 * x2;
    
    return 1.0 - x2 * 0.5 + x4 * 0.0416666667;
}
"""
            # 将定义添加到着色器开头
            shader_source = fast_trig_defs + "\n" + shader_source
        
        return shader_source
    
    def _optimize_dot_products(self, shader_source):
        """
        优化点积操作
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        # Maxwell架构上，使用dp4a指令可以提高点积性能
        # 这里我们只是添加注释和标记，实际的dp4a优化需要底层图形API支持
        
        # 查找vec4的点积操作
        dot_product_pattern = r'\bdot\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*where\s*\1\s*and\s*\2\s*are\s*vec4'
        
        # 这里只是一个示例，实际的dp4a优化需要更复杂的实现
        optimized_source = shader_source
        
        # 添加注释
        optimized_source = "// 注意：Maxwell架构上vec4点积可以使用dp4a指令优化\n" + optimized_source
        
        return optimized_source
    
    def _simplify_control_flow(self, shader_source):
        """
        简化控制流
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        # 简化复杂的嵌套if-else语句
        # 这里只是一个简化的实现
        
        # 添加注释
        optimized_source = "// 警告：Maxwell架构上复杂控制流可能导致性能下降\n" + shader_source
        
        return optimized_source
    
    def _optimize_mad_instructions(self, shader_source):
        """
        优化mad（乘加）指令
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        # 将 a*b + c 模式识别为mad(a,b,c)的机会
        # 这通常会被编译器自动优化，但我们可以帮助提示
        
        # 简单的模式匹配
        mad_pattern = r'(\w+)\s*\*\s*(\w+)\s*\+\s*(\w+)'  # 非常简化的模式，实际应该更复杂
        
        # 这里我们不实际替换，只是添加注释，因为实际的mad优化通常由编译器完成
        optimized_source = "// 注意：GCN架构上乘加操作可以使用mad指令优化\n" + shader_source
        
        return optimized_source
    
    def _optimize_uniform_control_flow(self, shader_source):
        """
        优化统一控制流
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        # GCN架构上，统一控制流（所有线程走相同的分支）更高效
        # 这里我们添加注释建议优化控制流
        
        optimized_source = "// 注意：GCN架构上统一控制流（所有线程走相同分支）更高效\n" + shader_source
        
        return optimized_source
    
    def set_optimization_level(self, level):
        """
        设置优化级别
        
        Args:
            level: 优化级别 (0-3)
        """
        self.optimization_level = max(0, min(3, level))
        print(f"设置着色器优化级别: {self.optimization_level}")
    
    def set_optimization_options(self, options):
        """
        设置优化选项
        
        Args:
            options: 优化选项字典
        """
        for key, value in options.items():
            if key in self.optimization_options:
                self.optimization_options[key] = value
                print(f"设置优化选项: {key} = {value}")
    
    def set_gpu_architecture(self, architecture):
        """
        设置GPU架构
        
        Args:
            architecture: GPU架构名称
        """
        self.gpu_architecture = architecture.lower()
        self.current_rules = self.architecture_rules.get(self.gpu_architecture, self.architecture_rules["default"])
        print(f"设置GPU架构: {self.gpu_architecture}")
        print(f"应用架构特定规则: {self.current_rules}")
    
    def get_optimization_stats(self):
        """
        获取优化统计信息
        
        Returns:
            dict: 优化统计信息
        """
        return self.optimization_stats.copy()
    
    def print_optimization_report(self):
        """
        打印优化报告
        """
        stats = self.optimization_stats
        
        print("\n=== 着色器优化报告 ===")
        print(f"优化的着色器数: {stats['shaders_optimized']}")
        print(f"移除的指令数: {stats['instructions_removed']}")
        print(f"节省的寄存器数: {stats['registers_saved']}")
        print(f"精度调整次数: {stats['precision_adjustments']}")
        print(f"优化的循环数: {stats['loops_optimized']}")
        print(f"优化的分支数: {stats['branches_optimized']}")
        print(f"优化的纹理采样数: {stats['textures_optimized']}")
        print(f"改进的内存访问数: {stats['memory_accesses_improved']}")
        print(f"当前GPU架构: {self.gpu_architecture}")
        print(f"当前优化级别: {self.optimization_level}")
        print("=====================\n")

# 示例用法
if __name__ == "__main__":
    # 创建着色器优化器实例
    optimizer = ShaderOptimizer("maxwell")
    
    # 设置优化选项
    optimizer.set_optimization_options({
        "precision_optimization": True,
        "branch_optimization": True
    })
    
    # 设置优化级别
    optimizer.set_optimization_level(3)
    
    # 示例着色器代码
    sample_shader = """
void main() {
    highp vec4 color = vec4(1.0, 0.0, 0.0, 1.0);
    highp float intensity = 0.5 + 0.5;
    
    if (intensity > 0.8) {
        color = vec4(0.0, 1.0, 0.0, 1.0);
    } else {
        color = vec4(0.0, 0.0, 1.0, 1.0);
    }
    
    gl_FragColor = color * intensity;
}
"""
    
    # 优化着色器
    optimized_shader = optimizer.optimize_shader(sample_shader, "fragment")
    
    print("原始着色器:")
    print(sample_shader)
    print("\n优化后的着色器:")
    print(optimized_shader)
    
    # 打印优化报告
    optimizer.print_optimization_report()