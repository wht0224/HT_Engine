import numpy as np
from .EffectBase import EffectBase, GPUArchitecture, EffectQuality
from .FastApproximateGlobalIllumination import FastApproximateGlobalIllumination
from .ScreenspaceReflections import ScreenspaceReflections
from .AmbientOcclusion import AmbientOcclusion
from .Bloom import Bloom
from .ColorGrading import ColorGrading
from .DepthOfField import DepthOfField
from .MotionBlur import MotionBlur
from .FidelityFXSuperResolution import FidelityFXSuperResolution
from .VolumetricLight import VolumetricLight
from .PathTracer import PathTracer

class EffectManager:
    """针对低端GPU优化的特效管理器"""
    
    def __init__(self, renderer):
        """
        初始化特效管理器
        
        参数:
        - renderer: 渲染器实例
        """
        self.renderer = renderer
        self.gpu_architecture = renderer.gpu_architecture if hasattr(renderer, 'gpu_architecture') else GPUArchitecture.OTHER
        self.quality_level = renderer.quality_level if hasattr(renderer, 'quality_level') else EffectQuality.LOW
        
        # 特效列表
        self.effects = {
            'path_tracer': PathTracer(self.gpu_architecture, self.quality_level),
            'fagi': FastApproximateGlobalIllumination(self.gpu_architecture, self.quality_level),
            'ao': AmbientOcclusion(self.gpu_architecture, self.quality_level),
            'ssr': ScreenspaceReflections(self.gpu_architecture, self.quality_level),
            'volumetric_light': VolumetricLight(self.gpu_architecture, self.quality_level),
            'bloom': Bloom(self.gpu_architecture, self.quality_level),
            'dof': DepthOfField(self.gpu_architecture, self.quality_level),
            'motion_blur': MotionBlur(self.gpu_architecture, self.quality_level),
            'color_grading': ColorGrading(self.gpu_architecture, self.quality_level),
            'fsr': FidelityFXSuperResolution(self.gpu_architecture, self.quality_level)
        }
        
        # 效果执行顺序
        self.execution_order = ['ao', 'fagi', 'ssr', 'volumetric_light', 'bloom', 'dof', 'motion_blur', 'color_grading', 'fsr']
        
        # 性能预算(ms)
        self.performance_budget = {
            EffectQuality.LOW: 5.0,  # GTX 750Ti的预算
            EffectQuality.MEDIUM: 8.0,  # RX 580的预算
            EffectQuality.HIGH: 15.0  # 高端GPU的预算
        }
        
        # 初始化所有效果
        self._initialize_effects()
        
        # 根据性能预算自动优化效果组合
        self.optimize_effects_for_performance()
    
    def _initialize_effects(self):
        """初始化所有效果"""
        for effect_name, effect in self.effects.items():
            try:
                effect.initialize(self.renderer)
                print(f"Effect '{effect_name}' initialized successfully")
            except Exception as e:
                print(f"Failed to initialize effect '{effect_name}': {e}")
                effect.is_enabled = False
    
    def update(self, delta_time):
        """
        更新所有启用的效果
        
        参数:
        - delta_time: 帧时间间隔
        """
        for effect_name in self.execution_order:
            effect = self.effects[effect_name]
            if effect.is_enabled:
                try:
                    effect.update(delta_time)
                except Exception as e:
                    print(f"Error updating effect '{effect_name}': {e}")
    
    def render(self, input_texture):
        """
        按顺序渲染所有启用的效果
        
        参数:
        - input_texture: 初始输入纹理
        
        返回:
        - 处理后的最终纹理
        """
        current_texture = input_texture
        
        for effect_name in self.execution_order:
            effect = self.effects[effect_name]
            if effect.is_enabled:
                try:
                    # 渲染当前效果并将结果传递给下一个效果
                    current_texture = effect.render(current_texture)
                except Exception as e:
                    print(f"Error rendering effect '{effect_name}': {e}")
                    # 发生错误时跳过该效果，使用上一个纹理继续
        
        return current_texture
    
    def optimize_effects_for_performance(self):
        """根据性能预算自动优化效果组合"""
        available_budget = self.performance_budget[self.quality_level]
        
        # 计算当前启用效果的总性能开销
        total_cost = sum(effect.get_performance_impact() for effect in self.effects.values() if effect.is_enabled)
        
        # 如果总开销超过预算，降低效果质量或禁用部分效果
        if total_cost > available_budget:
            self._reduce_performance_impact(total_cost - available_budget)
    
    def _reduce_performance_impact(self, over_budget):
        """
        减少性能开销以符合预算
        
        参数:
        - over_budget: 超出预算的毫秒数
        """
        # 按性能开销从高到低排序效果
        effects_by_cost = sorted(
            [(name, effect) for name, effect in self.effects.items() if effect.is_enabled],
            key=lambda x: x[1].get_performance_impact(),
            reverse=True
        )
        
        current_over_budget = over_budget
        
        # 尝试降低效果质量或禁用效果
        for effect_name, effect in effects_by_cost:
            if current_over_budget <= 0:
                break
            
            # 先尝试降低质量
            if effect.quality_level.value > EffectQuality.LOW.value:
                original_quality = effect.quality_level
                original_cost = effect.get_performance_impact()
                
                # 降低一个质量等级
                new_quality = EffectQuality(original_quality.value - 1)
                effect.adjust_quality(new_quality)
                
                new_cost = effect.get_performance_impact()
                saved_cost = original_cost - new_cost
                current_over_budget -= saved_cost
                
                print(f"Reduced {effect_name} quality to {new_quality.name}, saved {saved_cost:.2f}ms")
            
            # 如果仍然超出预算，禁用该效果
            if current_over_budget > 0:
                effect.is_enabled = False
                saved_cost = effect.get_performance_impact()
                current_over_budget -= saved_cost
                print(f"Disabled {effect_name}, saved {saved_cost:.2f}ms")
    
    def set_effect_quality(self, effect_name, quality_level):
        """
        设置特定效果的质量级别
        
        参数:
        - effect_name: 效果名称
        - quality_level: 质量级别
        """
        if effect_name in self.effects:
            self.effects[effect_name].adjust_quality(quality_level)
            # 更新后重新优化性能
            self.optimize_effects_for_performance()
            return True
        return False
    
    def toggle_effect(self, effect_name, enable=None):
        """
        切换效果的启用状态
        
        参数:
        - effect_name: 效果名称
        - enable: 如果为None则切换当前状态，否则设置为指定状态
        """
        if effect_name in self.effects:
            if enable is None:
                self.effects[effect_name].is_enabled = not self.effects[effect_name].is_enabled
            else:
                self.effects[effect_name].is_enabled = enable
            
            # 更新后重新优化性能
            self.optimize_effects_for_performance()
            return True
        return False
    
    def set_global_quality(self, quality_level):
        """
        设置全局效果质量级别
        
        参数:
        - quality_level: 质量级别
        """
        self.quality_level = quality_level
        
        # 更新所有效果的质量级别
        for effect in self.effects.values():
            effect.adjust_quality(quality_level)
        
        # 根据GPU架构进行额外优化
        self._optimize_for_gpu_architecture()
        
        # 重新优化性能
        self.optimize_effects_for_performance()
    
    def _optimize_for_gpu_architecture(self):
        """根据GPU架构自动调整效果设置"""
        # 所有架构默认禁用路径追踪
        self.effects['path_tracer'].is_enabled = False
        
        # NVIDIA Maxwell (GTX 750Ti)
        if self.gpu_architecture == GPUArchitecture.NVIDIA_MAXWELL:
            if self.quality_level == EffectQuality.LOW:
                # 低端Maxwell禁用昂贵效果，启用FSR性能模式
                self.effects['ssr'].is_enabled = False
                self.effects['dof'].is_enabled = False
                self.effects['motion_blur'].is_enabled = False
                self.effects['volumetric_light'].is_enabled = False  # 低端Maxwell禁用体积光
                self.effects['path_tracer'].is_enabled = False  # 低端Maxwell禁用路径追踪
                self.effects['fagi'].adjust_quality(EffectQuality.LOW)
                self.effects['ao'].adjust_quality(EffectQuality.LOW)
                self.effects['bloom'].adjust_quality(EffectQuality.LOW)
                self.effects['color_grading'].adjust_quality(EffectQuality.LOW)
                self.effects['fsr'].is_enabled = True
                self.effects['fsr'].set_quality_mode("performance")
            elif self.quality_level == EffectQuality.MEDIUM:
                # 中等质量下禁用SSR和DOF，启用FSR平衡模式
                self.effects['ssr'].is_enabled = False
                self.effects['dof'].is_enabled = False
                self.effects['motion_blur'].is_enabled = True
                self.effects['motion_blur'].adjust_quality(EffectQuality.LOW)
                self.effects['volumetric_light'].is_enabled = True  # 中等质量下启用简化体积光
                self.effects['volumetric_light'].adjust_quality(EffectQuality.LOW)
                self.effects['path_tracer'].is_enabled = False  # 中等质量Maxwell也禁用路径追踪
                self.effects['fagi'].adjust_quality(EffectQuality.LOW)
                self.effects['ao'].adjust_quality(EffectQuality.LOW)
                self.effects['bloom'].adjust_quality(EffectQuality.MEDIUM)
                self.effects['color_grading'].adjust_quality(EffectQuality.MEDIUM)
                self.effects['fsr'].is_enabled = True
                self.effects['fsr'].set_quality_mode("balanced")
        
        # AMD GCN (RX 580)
        elif self.gpu_architecture == GPUArchitecture.AMD_GCN:
            if self.quality_level == EffectQuality.LOW:
                # 低端GCN禁用SSR和DOF，启用FSR平衡模式
                self.effects['ssr'].is_enabled = False
                self.effects['dof'].is_enabled = False
                self.effects['motion_blur'].is_enabled = False
                self.effects['volumetric_light'].is_enabled = False  # 低端GCN禁用体积光
                self.effects['path_tracer'].is_enabled = False  # 低端GCN禁用路径追踪
                self.effects['fagi'].adjust_quality(EffectQuality.LOW)
                self.effects['ao'].adjust_quality(EffectQuality.LOW)
                self.effects['bloom'].adjust_quality(EffectQuality.LOW)
                self.effects['color_grading'].adjust_quality(EffectQuality.LOW)
                self.effects['fsr'].is_enabled = True
                self.effects['fsr'].set_quality_mode("balanced")
            elif self.quality_level == EffectQuality.MEDIUM:
                # 中等质量下可以启用SSR和DOF低质量版本，启用FSR质量模式
                self.effects['ssr'].is_enabled = True
                self.effects['ssr'].adjust_quality(EffectQuality.LOW)  # 即使全局是中等，SSR也保持低质量
                self.effects['dof'].is_enabled = True
                self.effects['dof'].adjust_quality(EffectQuality.LOW)
                self.effects['motion_blur'].is_enabled = True
                self.effects['motion_blur'].adjust_quality(EffectQuality.LOW)
                self.effects['volumetric_light'].is_enabled = True  # 中等质量下启用体积光
                self.effects['volumetric_light'].adjust_quality(EffectQuality.MEDIUM)
                self.effects['path_tracer'].is_enabled = False  # 中等质量GCN也禁用路径追踪
                self.effects['fagi'].adjust_quality(EffectQuality.MEDIUM)
                self.effects['ao'].adjust_quality(EffectQuality.MEDIUM)
                self.effects['bloom'].adjust_quality(EffectQuality.MEDIUM)
                self.effects['color_grading'].adjust_quality(EffectQuality.MEDIUM)
                self.effects['fsr'].is_enabled = True
                self.effects['fsr'].set_quality_mode("quality")
    
    def get_performance_stats(self):
        """
        获取效果性能统计信息
        
        返回:
        - 包含各效果性能数据的字典
        """
        stats = {
            'total_budget': self.performance_budget[self.quality_level],
            'total_usage': 0.0,
            'effects': {}
        }
        
        for effect_name, effect in self.effects.items():
            cost = effect.get_performance_impact() if effect.is_enabled else 0.0
            stats['effects'][effect_name] = {
                'enabled': effect.is_enabled,
                'cost': cost,
                'quality': effect.quality_level.name
            }
            stats['total_usage'] += cost
        
        stats['budget_status'] = stats['total_usage'] <= stats['total_budget']
        stats['over_budget'] = max(0.0, stats['total_usage'] - stats['total_budget'])
        
        return stats
    
    def add_custom_effect(self, effect_name, effect):
        """
        添加自定义效果
        
        参数:
        - effect_name: 效果名称
        - effect: EffectBase的子类实例
        """
        if isinstance(effect, EffectBase):
            self.effects[effect_name] = effect
            effect.initialize(self.renderer)
            
            # 添加到执行顺序末尾
            if effect_name not in self.execution_order:
                self.execution_order.append(effect_name)
            
            # 重新优化性能
            self.optimize_effects_for_performance()
            return True
        return False
    
    def remove_effect(self, effect_name):
        """
        移除效果
        
        参数:
        - effect_name: 效果名称
        """
        if effect_name in self.effects:
            del self.effects[effect_name]
            
            # 从执行顺序中移除
            if effect_name in self.execution_order:
                self.execution_order.remove(effect_name)
            
            return True
        return False