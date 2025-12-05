# -*- coding: utf-8 -*-
"""
低端GPU渲染引擎视觉质量验证工具
用于比较渲染结果与参考图像，评估优化后的视觉质量
"""

import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from Engine.Math import Vector3


class QualityValidator:
    """视觉质量验证管理器"""
    
    def __init__(self, engine=None):
        """初始化质量验证工具
        
        Args:
            engine: 渲染引擎实例（可选）
        """
        self.engine = engine
        
        # 参考图像和测试图像
        self.reference_image = None
        self.test_image = None
        
        # 验证结果
        self.validation_results = {}
        
        # 质量阈值设置
        self.quality_thresholds = {
            "psnr": 30.0,  # PSNR阈值（dB）
            "ssim": 0.9,   # SSIM阈值
            "mse": 10.0,   # MSE阈值
            "lpips": 0.15  # LPIPS感知相似度阈值（模拟值）
        }
        
        # 错误图生成设置
        self.error_map_settings = {
            "enhance_edges": True,  # 是否增强边缘差异
            "color_scheme": "jet",  # 色彩映射方案
            "normalize": True,      # 是否归一化错误值
            "threshold": 0.1        # 最小错误阈值
        }
        
        # 测试场景配置
        self.test_scenes = {
            "standard_scene": {
                "reference_path": None,
                "description": "标准测试场景",
                "complexity": 5  # 复杂度等级1-10
            },
            "high_detail_scene": {
                "reference_path": None,
                "description": "高细节测试场景",
                "complexity": 8
            },
            "low_light_scene": {
                "reference_path": None,
                "description": "低光照测试场景",
                "complexity": 7
            },
            "motion_blur_scene": {
                "reference_path": None,
                "description": "运动模糊测试场景",
                "complexity": 6
            }
        }
        
        # 针对不同GPU的质量期望值
        self.target_quality = {
            "GTX_750Ti": {
                "min_psnr": 28.0,
                "min_ssim": 0.85,
                "acceptable_error_percentage": 5.0  # 允许5%像素有明显差异
            },
            "RX_580": {
                "min_psnr": 32.0,
                "min_ssim": 0.92,
                "acceptable_error_percentage": 3.0  # 允许3%像素有明显差异
            }
        }
    
    def load_reference_image(self, image_path):
        """加载参考图像
        
        Args:
            image_path: 参考图像路径
            
        Returns:
            bool: 加载是否成功
        """
        if not os.path.exists(image_path):
            print(f"错误: 找不到参考图像 {image_path}")
            return False
        
        try:
            # 使用OpenCV加载图像，转换为RGB格式
            self.reference_image = cv2.imread(image_path)
            if self.reference_image is None:
                print(f"错误: 无法读取参考图像 {image_path}")
                return False
            
            # 转换BGR到RGB
            self.reference_image = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
            print(f"成功加载参考图像: {image_path}")
            return True
        except Exception as e:
            print(f"加载参考图像时出错: {e}")
            return False
    
    def load_test_image(self, image_path):
        """加载测试图像
        
        Args:
            image_path: 测试图像路径
            
        Returns:
            bool: 加载是否成功
        """
        if not os.path.exists(image_path):
            print(f"错误: 找不到测试图像 {image_path}")
            return False
        
        try:
            # 使用OpenCV加载图像，转换为RGB格式
            self.test_image = cv2.imread(image_path)
            if self.test_image is None:
                print(f"错误: 无法读取测试图像 {image_path}")
                return False
            
            # 转换BGR到RGB
            self.test_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2RGB)
            print(f"成功加载测试图像: {image_path}")
            return True
        except Exception as e:
            print(f"加载测试图像时出错: {e}")
            return False
    
    def validate_images(self, reference_path=None, test_path=None):
        """验证两张图像的质量差异
        
        Args:
            reference_path: 参考图像路径（可选，如已加载则不需提供）
            test_path: 测试图像路径（可选，如已加载则不需提供）
            
        Returns:
            dict: 验证结果
        """
        # 确保图像已加载
        if reference_path is not None:
            if not self.load_reference_image(reference_path):
                return None
        
        if test_path is not None:
            if not self.load_test_image(test_path):
                return None
        
        if self.reference_image is None or self.test_image is None:
            print("错误: 请先加载参考图像和测试图像")
            return None
        
        # 确保图像尺寸相同
        if self.reference_image.shape != self.test_image.shape:
            print("警告: 图像尺寸不匹配，调整测试图像尺寸以匹配参考图像")
            self.test_image = cv2.resize(self.test_image, 
                                        (self.reference_image.shape[1], 
                                         self.reference_image.shape[0]))
        
        # 初始化结果字典
        results = {
            "reference_shape": self.reference_image.shape,
            "test_shape": self.test_image.shape,
            "metrics": {},
            "quality_passed": False,
            "detailed_analysis": {},
            "suggestions": []
        }
        
        # 计算各种质量指标
        results["metrics"] = self._calculate_quality_metrics()
        
        # 详细分析
        results["detailed_analysis"] = self._perform_detailed_analysis()
        
        # 评估是否通过质量测试
        results["quality_passed"] = self._evaluate_quality_pass(results["metrics"])
        
        # 生成优化建议
        results["suggestions"] = self._generate_optimization_suggestions(results)
        
        # 保存结果
        self.validation_results = results
        
        # 输出验证报告
        self._print_validation_report(results)
        
        return results
    
    def _calculate_quality_metrics(self):
        """计算各种质量指标
        
        Returns:
            dict: 质量指标结果
        """
        metrics = {}
        
        # 确保图像是浮点型（0-1范围）用于计算
        ref_float = self.reference_image.astype(np.float32) / 255.0
        test_float = self.test_image.astype(np.float32) / 255.0
        
        # 计算MSE（均方误差）
        mse = np.mean((ref_float - test_float) ** 2)
        metrics["mse"] = mse
        
        # 计算PSNR（峰值信噪比）
        # 使用skimage的PSNR实现，也可以手动计算
        try:
            psnr = psnr_skimage(self.reference_image, self.test_image)
        except:
            # 手动计算PSNR作为备选
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        metrics["psnr"] = psnr
        
        # 计算SSIM（结构相似性指数）
        # 转换为灰度图进行SSIM计算
        ref_gray = cv2.cvtColor(self.reference_image, cv2.COLOR_RGB2GRAY)
        test_gray = cv2.cvtColor(self.test_image, cv2.COLOR_RGB2GRAY)
        
        try:
            ssim_value = ssim(ref_gray, test_gray)
        except:
            # 简单的SSIM近似计算作为备选
            ref_mean = np.mean(ref_gray)
            test_mean = np.mean(test_gray)
            cov = np.mean((ref_gray - ref_mean) * (test_gray - test_mean))
            ref_var = np.mean((ref_gray - ref_mean) ** 2)
            test_var = np.mean((test_gray - test_mean) ** 2)
            
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            # 计算SSIM值
            numerator = (2 * ref_mean * test_mean + C1) * (2 * cov + C2)
            denominator = (ref_mean ** 2 + test_mean ** 2 + C1) * (ref_var + test_var + C2)
            ssim_value = numerator / denominator
        
        metrics["ssim"] = ssim_value
        
        # 计算色彩差异
        color_diff = np.mean(np.abs(ref_float - test_float), axis=2)
        metrics["average_color_diff"] = np.mean(color_diff)
        metrics["max_color_diff"] = np.max(color_diff)
        
        # 计算像素错误百分比（超过阈值的像素比例）
        threshold = self.error_map_settings["threshold"]
        error_pixels = np.sum(color_diff > threshold)
        total_pixels = color_diff.size
        metrics["error_pixel_percentage"] = (error_pixels / total_pixels) * 100
        
        # 计算亮度差异
        ref_luminance = 0.299 * ref_float[:,:,0] + 0.587 * ref_float[:,:,1] + 0.114 * ref_float[:,:,2]
        test_luminance = 0.299 * test_float[:,:,0] + 0.587 * test_float[:,:,1] + 0.114 * test_float[:,:,2]
        luminance_diff = np.mean(np.abs(ref_luminance - test_luminance))
        metrics["luminance_diff"] = luminance_diff
        
        # 模拟LPIPS感知相似度（简化版本）
        # 基于边缘保持和颜色相似度的组合
        edges_ref = cv2.Canny(self.reference_image, 50, 150)
        edges_test = cv2.Canny(self.test_image, 50, 150)
        edge_similarity = np.mean(edges_ref == edges_test) / 255.0
        
        # 组合多种指标作为感知相似度的近似
        perceptual_similarity = 0.4 * ssim_value + 0.3 * edge_similarity + 0.3 * (1 - metrics["average_color_diff"])
        metrics["perceptual_similarity"] = perceptual_similarity
        
        return metrics
    
    def _perform_detailed_analysis(self):
        """执行详细的图像质量分析
        
        Returns:
            dict: 详细分析结果
        """
        analysis = {
            "error_distribution": {},
            "problematic_regions": [],
            "edge_analysis": {},
            "color_analysis": {}
        }
        
        # 计算错误图
        ref_float = self.reference_image.astype(np.float32) / 255.0
        test_float = self.test_image.astype(np.float32) / 255.0
        error_map = np.mean(np.abs(ref_float - test_float), axis=2)
        
        # 分析错误分布
        error_levels = {
            "very_low": (0, 0.05),
            "low": (0.05, 0.1),
            "medium": (0.1, 0.2),
            "high": (0.2, 0.5),
            "very_high": (0.5, 1.0)
        }
        
        for level, (min_val, max_val) in error_levels.items():
            if level == "very_low":
                count = np.sum((error_map >= min_val) & (error_map < max_val))
            elif level == "very_high":
                count = np.sum((error_map >= min_val) & (error_map <= max_val))
            else:
                count = np.sum((error_map >= min_val) & (error_map < max_val))
            
            percentage = (count / error_map.size) * 100
            analysis["error_distribution"][level] = percentage
        
        # 检测问题区域（高误差区域）
        threshold = 0.2  # 较高误差阈值
        high_error_regions = error_map > threshold
        
        if np.any(high_error_regions):
            # 查找连通区域
            high_error_uint8 = (high_error_regions * 255).astype(np.uint8)
            contours, _ = cv2.findContours(high_error_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 分析每个轮廓
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 100:  # 忽略很小的区域
                    x, y, w, h = cv2.boundingRect(contour)
                    region_error = np.mean(error_map[y:y+h, x:x+w])
                    
                    analysis["problematic_regions"].append({
                        "id": i,
                        "position": (x, y),
                        "size": (w, h),
                        "area": area,
                        "average_error": region_error,
                        "description": self._describe_region(x, y, w, h)
                    })
        
        # 边缘分析
        edges_ref = cv2.Canny(self.reference_image, 50, 150)
        edges_test = cv2.Canny(self.test_image, 50, 150)
        
        edge_diff = np.abs(edges_ref.astype(float) - edges_test.astype(float)) / 255.0
        analysis["edge_analysis"]["edge_similarity"] = 1.0 - np.mean(edge_diff)
        analysis["edge_analysis"]["missing_edges"] = np.sum((edges_ref > 0) & (edges_test == 0))
        analysis["edge_analysis"]["extra_edges"] = np.sum((edges_ref == 0) & (edges_test > 0))
        
        # 颜色分析
        ref_mean_color = np.mean(self.reference_image, axis=(0, 1))
        test_mean_color = np.mean(self.test_image, axis=(0, 1))
        color_bias = np.abs(ref_mean_color - test_mean_color)
        
        analysis["color_analysis"]["reference_mean_color"] = ref_mean_color.tolist()
        analysis["color_analysis"]["test_mean_color"] = test_mean_color.tolist()
        analysis["color_analysis"]["color_bias"] = color_bias.tolist()
        
        # 检测颜色偏移的通道
        if color_bias[0] > color_bias[1] and color_bias[0] > color_bias[2]:
            analysis["color_analysis"]["dominant_bias"] = "红色通道"
        elif color_bias[1] > color_bias[0] and color_bias[1] > color_bias[2]:
            analysis["color_analysis"]["dominant_bias"] = "绿色通道"
        elif color_bias[2] > color_bias[0] and color_bias[2] > color_bias[1]:
            analysis["color_analysis"]["dominant_bias"] = "蓝色通道"
        else:
            analysis["color_analysis"]["dominant_bias"] = "均匀分布"
        
        return analysis
    
    def _describe_region(self, x, y, w, h):
        """描述图像区域的内容
        
        Args:
            x, y: 区域左上角坐标
            w, h: 区域宽度和高度
            
        Returns:
            str: 区域描述
        """
        # 提取区域像素
        region = self.reference_image[y:y+h, x:x+w]
        
        # 计算区域特性
        mean_brightness = np.mean(region) / 255.0
        color_std = np.std(region, axis=(0, 1))
        texture_energy = np.mean(cv2.Laplacian(region, cv2.CV_64F)**2)
        
        description = []
        
        # 亮度描述
        if mean_brightness < 0.3:
            description.append("暗区域")
        elif mean_brightness > 0.7:
            description.append("亮区域")
        else:
            description.append("中等亮度区域")
        
        # 颜色变化描述
        color_variation = np.mean(color_std) / 255.0
        if color_variation < 0.1:
            description.append("颜色均匀")
        elif color_variation > 0.3:
            description.append("颜色多变")
        
        # 纹理描述
        if texture_energy < 1000:
            description.append("平滑区域")
        elif texture_energy > 5000:
            description.append("纹理丰富区域")
        
        return "，".join(description)
    
    def _evaluate_quality_pass(self, metrics):
        """评估是否通过质量测试
        
        Args:
            metrics: 质量指标
            
        Returns:
            bool: 是否通过测试
        """
        # 确定目标GPU类型
        gpu_type = "GTX_750Ti"  # 默认值
        if self.engine and hasattr(self.engine, 'platform') and self.engine.platform:
            gpu_name = self.engine.platform.gpu_name.upper()
            if "RX 580" in gpu_name or "RX580" in gpu_name:
                gpu_type = "RX_580"
        
        # 获取对应的质量要求
        quality_requirements = self.target_quality.get(gpu_type, self.target_quality["GTX_750Ti"])
        
        # 检查关键指标
        psnr_passed = metrics["psnr"] >= quality_requirements["min_psnr"]
        ssim_passed = metrics["ssim"] >= quality_requirements["min_ssim"]
        error_percentage_passed = metrics["error_pixel_percentage"] <= quality_requirements["acceptable_error_percentage"]
        
        # 所有指标都需要通过
        return psnr_passed and ssim_passed and error_percentage_passed
    
    def _generate_optimization_suggestions(self, results):
        """生成优化建议
        
        Args:
            results: 验证结果
            
        Returns:
            list: 优化建议列表
        """
        suggestions = []
        metrics = results["metrics"]
        analysis = results["detailed_analysis"]
        
        # 基于PSNR的建议
        if metrics["psnr"] < self.quality_thresholds["psnr"]:
            suggestions.append("整体图像质量偏低，考虑提高渲染分辨率或降低压缩率")
        
        # 基于SSIM的建议
        if metrics["ssim"] < self.quality_thresholds["ssim"]:
            suggestions.append("图像结构相似性不足，检查是否有几何渲染错误或纹理映射问题")
        
        # 基于错误分布的建议
        high_error_percentage = analysis["error_distribution"].get("high", 0) + analysis["error_distribution"].get("very_high", 0)
        if high_error_percentage > 5.0:
            suggestions.append("发现大面积高误差区域，可能是光照计算或着色器实现有误")
        
        # 边缘分析建议
        if analysis["edge_analysis"]["edge_similarity"] < 0.85:
            suggestions.append("边缘细节丢失较多，考虑调整抗锯齿设置或边缘锐化参数")
            
            if analysis["edge_analysis"]["missing_edges"] > analysis["edge_analysis"]["extra_edges"]:
                suggestions.append("边缘细节被过度模糊，可能是抗锯齿或后处理效果过强")
            else:
                suggestions.append("出现额外噪点或伪影，可能是着色器精度问题或数值不稳定")
        
        # 颜色分析建议
        color_bias = np.mean(analysis["color_analysis"]["color_bias"])
        if color_bias > 15.0:
            dominant_bias = analysis["color_analysis"]["dominant_bias"]
            suggestions.append(f"存在明显的色彩偏移，主要集中在{dominant_bias}，检查颜色空间转换或渲染管线中的颜色处理")
        
        # 针对问题区域的建议
        if analysis["problematic_regions"]:
            # 分析问题区域的共性
            texture_regions = sum("纹理丰富" in r["description"] for r in analysis["problematic_regions"])
            dark_regions = sum("暗区域" in r["description"] for r in analysis["problematic_regions"])
            
            if texture_regions > len(analysis["problematic_regions"]) * 0.5:
                suggestions.append("纹理丰富区域质量问题较多，考虑提高纹理采样质量或减少压缩")
            
            if dark_regions > len(analysis["problematic_regions"]) * 0.5:
                suggestions.append("暗区域存在较多渲染错误，检查光照计算精度或考虑使用对数空间渲染")
        
        # 性能与质量平衡建议
        if results["quality_passed"]:
            suggestions.append("当前渲染质量已达标，但可考虑以下优化以提高性能：")
            
            # 如果各项指标远高于阈值，建议降低某些设置以提高性能
            if metrics["ssim"] > self.quality_thresholds["ssim"] + 0.05:
                suggestions.append("- 可适当降低纹理分辨率以提高性能")
            
            if metrics["psnr"] > self.quality_thresholds["psnr"] + 5.0:
                suggestions.append("- 考虑使用更高效的着色器变体以减少GPU计算负担")
        
        return suggestions
    
    def _print_validation_report(self, results):
        """打印验证报告
        
        Args:
            results: 验证结果
        """
        print("\n===== 视觉质量验证报告 =====")
        
        # 基本信息
        print(f"参考图像尺寸: {results['reference_shape'][1]}x{results['reference_shape'][0]}")
        print(f"测试图像尺寸: {results['test_shape'][1]}x{results['test_shape'][0]}")
        print(f"质量测试结果: {'通过' if results['quality_passed'] else '未通过'}")
        
        # 质量指标
        print("\n质量指标:")
        metrics = results["metrics"]
        print(f"PSNR: {metrics['psnr']:.2f} dB (阈值: {self.quality_thresholds['psnr']} dB)")
        print(f"SSIM: {metrics['ssim']:.3f} (阈值: {self.quality_thresholds['ssim']})")
        print(f"MSE: {metrics['mse']:.2f} (阈值: {self.quality_thresholds['mse']})")
        print(f"平均颜色差异: {metrics['average_color_diff']*100:.2f}%")
        print(f"像素错误百分比: {metrics['error_pixel_percentage']:.2f}%")
        print(f"感知相似度: {metrics['perceptual_similarity']:.3f}")
        
        # 错误分布
        print("\n错误分布:")
        error_dist = results["detailed_analysis"]["error_distribution"]
        for level, percentage in error_dist.items():
            print(f"  {level}: {percentage:.2f}%")
        
        # 问题区域
        problem_regions = results["detailed_analysis"]["problematic_regions"]
        if problem_regions:
            print(f"\n发现 {len(problem_regions)} 个主要问题区域:")
            for region in problem_regions[:3]:  # 只显示前3个问题区域
                print(f"  区域 {region['id']}: 位置({region['position'][0]}, {region['position'][1]}), 大小{region['size'][0]}x{region['size'][1]}, 平均误差{region['average_error']:.3f}, {region['description']}")
        
        # 颜色分析
        color_analysis = results["detailed_analysis"]["color_analysis"]
        print(f"\n颜色分析:")
        print(f"  参考图像平均颜色: R{color_analysis['reference_mean_color'][0]:.1f}, G{color_analysis['reference_mean_color'][1]:.1f}, B{color_analysis['reference_mean_color'][2]:.1f}")
        print(f"  测试图像平均颜色: R{color_analysis['test_mean_color'][0]:.1f}, G{color_analysis['test_mean_color'][1]:.1f}, B{color_analysis['test_mean_color'][2]:.1f}")
        print(f"  颜色偏移: {np.mean(color_analysis['color_bias']):.1f}, 主要集中在{color_analysis['dominant_bias']}")
        
        # 优化建议
        print("\n优化建议:")
        for i, suggestion in enumerate(results["suggestions"], 1):
            print(f"  {i}. {suggestion}")
        
        print("=============================\n")
    
    def generate_error_map(self, output_path=None):
        """生成错误图用于可视化比较
        
        Args:
            output_path: 输出路径，如不提供则只返回图像数据
            
        Returns:
            numpy.ndarray: 错误图
        """
        if self.reference_image is None or self.test_image is None:
            print("错误: 请先加载参考图像和测试图像")
            return None
        
        # 确保图像尺寸相同
        if self.reference_image.shape != self.test_image.shape:
            test_resized = cv2.resize(self.test_image, 
                                     (self.reference_image.shape[1], 
                                      self.reference_image.shape[0]))
        else:
            test_resized = self.test_image
        
        # 计算每个通道的差异
        bgr_ref = cv2.cvtColor(self.reference_image, cv2.COLOR_RGB2BGR)
        bgr_test = cv2.cvtColor(test_resized, cv2.COLOR_RGB2BGR)
        
        # 计算绝对值差异
        diff = cv2.absdiff(bgr_ref, bgr_test)
        
        # 如果设置了边缘增强
        if self.error_map_settings["enhance_edges"]:
            # 使用拉普拉斯算子检测边缘
            edges_ref = cv2.Laplacian(cv2.cvtColor(self.reference_image, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
            edges_test = cv2.Laplacian(cv2.cvtColor(test_resized, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
            edge_diff = cv2.convertScaleAbs(cv2.absdiff(edges_ref, edges_test))
            
            # 将边缘差异添加到色彩差异中
            edge_diff_colored = cv2.merge([edge_diff, edge_diff, edge_diff])
            diff = cv2.addWeighted(diff, 0.7, edge_diff_colored, 0.3, 0)
        
        # 归一化差异图像
        if self.error_map_settings["normalize"]:
            min_val = np.min(diff)
            max_val = np.max(diff)
            if max_val > min_val:
                diff = ((diff - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        # 应用色彩映射
        color_scheme = self.error_map_settings["color_scheme"]
        if color_scheme == "jet":
            error_map = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        elif color_scheme == "hot":
            error_map = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
        elif color_scheme == "cool":
            error_map = cv2.applyColorMap(diff, cv2.COLORMAP_COOL)
        else:
            error_map = diff
        
        # 保存错误图
        if output_path:
            try:
                cv2.imwrite(output_path, error_map)
                print(f"错误图已保存至: {output_path}")
            except Exception as e:
                print(f"保存错误图时出错: {e}")
        
        return error_map
    
    def create_comparison_image(self, output_path=None, layout="horizontal"):
        """创建包含参考图像、测试图像和差异图的比较图像
        
        Args:
            output_path: 输出路径
            layout: 布局方式 ("horizontal" 或 "vertical")
            
        Returns:
            numpy.ndarray: 比较图像
        """
        if self.reference_image is None or self.test_image is None:
            print("错误: 请先加载参考图像和测试图像")
            return None
        
        # 确保图像尺寸相同
        if self.reference_image.shape != self.test_image.shape:
            test_resized = cv2.resize(self.test_image, 
                                     (self.reference_image.shape[1], 
                                      self.reference_image.shape[0]))
        else:
            test_resized = self.test_image
        
        # 转换为BGR格式用于OpenCV处理
        ref_bgr = cv2.cvtColor(self.reference_image, cv2.COLOR_RGB2BGR)
        test_bgr = cv2.cvtColor(test_resized, cv2.COLOR_RGB2BGR)
        
        # 生成错误图
        error_map = self.generate_error_map()
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        thickness = 1
        
        # 在图像上添加文字标签
        cv2.putText(ref_bgr, "参考图像", (10, 30), font, font_scale, font_color, thickness)
        cv2.putText(test_bgr, "测试图像", (10, 30), font, font_scale, font_color, thickness)
        cv2.putText(error_map, "错误图", (10, 30), font, font_scale, font_color, thickness)
        
        # 根据布局组合图像
        if layout == "horizontal":
            comparison = np.hstack((ref_bgr, test_bgr, error_map))
        else:  # vertical
            comparison = np.vstack((ref_bgr, test_bgr, error_map))
        
        # 保存比较图像
        if output_path:
            try:
                cv2.imwrite(output_path, comparison)
                print(f"比较图像已保存至: {output_path}")
            except Exception as e:
                print(f"保存比较图像时出错: {e}")
        
        return comparison
    
    def compare_with_rtx4090_quality(self, test_image_path, reference_rtx_path):
        """将当前渲染质量与RTX 4090参考图像进行比较
        
        Args:
            test_image_path: 当前渲染图像路径
            reference_rtx_path: RTX 4090参考渲染图像路径
            
        Returns:
            dict: 比较结果
        """
        # 加载图像
        if not self.load_reference_image(reference_rtx_path):
            return None
        
        if not self.load_test_image(test_image_path):
            return None
        
        # 执行验证
        results = self.validate_images()
        if not results:
            return None
        
        # 添加与RTX 4090的比较评估
        quality_ratio = self._calculate_quality_ratio(results)
        results["rtx4090_comparison"] = {
            "quality_ratio": quality_ratio,
            "visual_fidelity": self._evaluate_visual_fidelity(quality_ratio),
            "key_differences": self._identify_key_differences(results)
        }
        
        # 打印RTX 4090比较报告
        self._print_rtx4090_comparison(results)
        
        return results
    
    def _calculate_quality_ratio(self, results):
        """计算质量比率
        
        Args:
            results: 验证结果
            
        Returns:
            float: 质量比率 (0-1)
        """
        metrics = results["metrics"]
        
        # 归一化各种指标到0-1范围
        # PSNR归一化（假设40dB是满分）
        psnr_norm = min(metrics["psnr"] / 40.0, 1.0)
        
        # SSIM已经是0-1范围
        ssim_norm = metrics["ssim"]
        
        # 平均颜色差异归一化
        color_diff_norm = max(1.0 - metrics["average_color_diff"], 0.0)
        
        # 错误像素百分比归一化
        error_pixel_norm = max(1.0 - (metrics["error_pixel_percentage"] / 100.0), 0.0)
        
        # 感知相似度
        perceptual_norm = metrics["perceptual_similarity"]
        
        # 加权平均
        weights = {
            "psnr": 0.2,
            "ssim": 0.3,
            "color_diff": 0.15,
            "error_pixel": 0.15,
            "perceptual": 0.2
        }
        
        quality_ratio = (
            psnr_norm * weights["psnr"] +
            ssim_norm * weights["ssim"] +
            color_diff_norm * weights["color_diff"] +
            error_pixel_norm * weights["error_pixel"] +
            perceptual_norm * weights["perceptual"]
        )
        
        return quality_ratio
    
    def _evaluate_visual_fidelity(self, quality_ratio):
        """评估视觉保真度等级
        
        Args:
            quality_ratio: 质量比率
            
        Returns:
            dict: 视觉保真度评估
        """
        if quality_ratio >= 0.95:
            return {
                "level": "极高保真度",
                "description": "几乎无法区分与RTX 4090渲染的差异",
                "suitable_for": "高端视觉呈现，专业展示"
            }
        elif quality_ratio >= 0.85:
            return {
                "level": "高保真度",
                "description": "仅在仔细观察下才能发现细微差异",
                "suitable_for": "游戏、高质量可视化"
            }
        elif quality_ratio >= 0.75:
            return {
                "level": "中等保真度",
                "description": "存在可察觉的差异，但整体视觉效果良好",
                "suitable_for": "大多数实时应用场景"
            }
        elif quality_ratio >= 0.6:
            return {
                "level": "基本保真度",
                "description": "有明显差异，但保持了基本的视觉特征",
                "suitable_for": "对性能要求极高的应用"
            }
        else:
            return {
                "level": "低保真度",
                "description": "差异显著，需要进一步优化",
                "suitable_for": "原型开发或极端性能受限场景"
            }
    
    def _identify_key_differences(self, results):
        """识别与RTX 4090参考图像的关键差异
        
        Args:
            results: 验证结果
            
        Returns:
            list: 关键差异列表
        """
        differences = []
        metrics = results["metrics"]
        analysis = results["detailed_analysis"]
        
        # 基于PSNR的差异
        if metrics["psnr"] < 30.0:
            differences.append("整体图像清晰度与RTX 4090参考有明显差距")
        
        # 基于SSIM的差异
        if metrics["ssim"] < 0.9:
            differences.append("图像结构细节保留不足，可能缺少精细纹理或几何细节")
        
        # 边缘差异
        if analysis["edge_analysis"]["edge_similarity"] < 0.85:
            if analysis["edge_analysis"]["missing_edges"] > analysis["edge_analysis"]["extra_edges"]:
                differences.append("边缘细节丢失较多，图像看起来更模糊")
            else:
                differences.append("存在额外的噪点或锯齿，抗锯齿效果不佳")
        
        # 颜色差异
        color_bias = np.mean(analysis["color_analysis"]["color_bias"])
        if color_bias > 20.0:
            differences.append(f"颜色表现与参考图像有明显偏差，主要在{analysis["color_analysis"]["dominant_bias"]}")
        
        # 问题区域分析
        if analysis["problematic_regions"]:
            texture_regions = sum("纹理丰富" in r["description"] for r in analysis["problematic_regions"])
            bright_regions = sum("亮区域" in r["description"] for r in analysis["problematic_regions"])
            
            if texture_regions > 0:
                differences.append("在复杂纹理区域表现较差，可能是纹理压缩或采样质量问题")
            
            if bright_regions > 0:
                differences.append("在高亮度区域存在明显差异，可能是HDR处理或曝光控制问题")
        
        return differences
    
    def _print_rtx4090_comparison(self, results):
        """打印与RTX 4090的比较报告
        
        Args:
            results: 包含RTX 4090比较的验证结果
        """
        rtx_comparison = results.get("rtx4090_comparison", {})
        
        print("\n===== 与RTX 4090渲染质量比较 =====")
        print(f"质量相似度: {rtx_comparison.get('quality_ratio', 0) * 100:.1f}%")
        
        fidelity = rtx_comparison.get('visual_fidelity', {})
        print(f"视觉保真度: {fidelity.get('level', '未知')}")
        print(f"描述: {fidelity.get('description', '无可用描述')}")
        print(f"适用场景: {fidelity.get('suitable_for', '无适用场景')}")
        
        key_differences = rtx_comparison.get('key_differences', [])
        if key_differences:
            print("\n主要差异:")
            for i, diff in enumerate(key_differences, 1):
                print(f"  {i}. {diff}")
        
        print("==============================\n")
    
    def save_validation_report(self, filename="validation_report.txt"):
        """保存验证报告到文件
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否保存成功
        """
        if not self.validation_results:
            print("没有可用的验证结果")
            return False
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("===== 低端GPU渲染引擎视觉质量验证报告 =====\n\n")
                
                # 基本信息
                f.write("基本信息:\n")
                f.write(f"参考图像尺寸: {self.validation_results['reference_shape'][1]}x{self.validation_results['reference_shape'][0]}\n")
                f.write(f"测试图像尺寸: {self.validation_results['test_shape'][1]}x{self.validation_results['test_shape'][0]}\n")
                f.write(f"质量测试结果: {'通过' if self.validation_results['quality_passed'] else '未通过'}\n\n")
                
                # 质量指标
                f.write("质量指标:\n")
                metrics = self.validation_results["metrics"]
                for key, value in metrics.items():
                    if key in self.quality_thresholds:
                        f.write(f"{key}: {value:.3f} (阈值: {self.quality_thresholds[key]})\n")
                    else:
                        f.write(f"{key}: {value:.3f}\n")
                
                # 错误分布
                f.write("\n错误分布:\n")
                error_dist = self.validation_results["detailed_analysis"]["error_distribution"]
                for level, percentage in error_dist.items():
                    f.write(f"  {level}: {percentage:.2f}%\n")
                
                # 优化建议
                f.write("\n优化建议:\n")
                for i, suggestion in enumerate(self.validation_results["suggestions"], 1):
                    f.write(f"  {i}. {suggestion}\n")
                
                # RTX 4090比较（如果有）
                if "rtx4090_comparison" in self.validation_results:
                    rtx_comparison = self.validation_results["rtx4090_comparison"]
                    f.write("\n===== 与RTX 4090渲染质量比较 =====\n")
                    f.write(f"质量相似度: {rtx_comparison.get('quality_ratio', 0) * 100:.1f}%\n")
                    
                    fidelity = rtx_comparison.get('visual_fidelity', {})
                    f.write(f"视觉保真度: {fidelity.get('level', '未知')}\n")
                    f.write(f"描述: {fidelity.get('description', '无可用描述')}\n")
                
            print(f"验证报告已保存至: {filename}")
            return True
        except Exception as e:
            print(f"保存验证报告时出错: {e}")
            return False