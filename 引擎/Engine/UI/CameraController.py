from Engine.Math import Vector3, Matrix4x4, Quaternion
import math

class CameraController:
    """相机控制器，支持多种相机控制方式"""
    
    # 相机模式枚举
    class Mode:
        FREE = 0  # 自由相机模式
        ORBIT = 1  # 轨道相机模式
        FIRST_PERSON = 2  # 第一人称相机模式
    
    # 鼠标按钮常量
    MOUSE_LEFT = 1
    MOUSE_MIDDLE = 2
    MOUSE_RIGHT = 3
    
    # 相机参数常量
    MIN_PITCH = -math.pi/2 + 0.01
    MAX_PITCH = math.pi/2 - 0.01
    MIN_ORBIT_DISTANCE = 0.1
    
    # 默认速度参数
    DEFAULT_MOVE_SPEED = 0.1
    DEFAULT_ROTATE_SPEED = 0.01
    DEFAULT_ZOOM_SPEED = 0.1
    
    def __init__(self, camera):
        self.camera = camera
        self.mode = self.Mode.FREE
        self.is_mouse_down = [False, False, False]  # 左键、中键、右键
        self.move_speed = self.DEFAULT_MOVE_SPEED
        self.rotate_speed = self.DEFAULT_ROTATE_SPEED
        self.zoom_speed = self.DEFAULT_ZOOM_SPEED
        self.is_ui_active = False
        
        # 轨道相机参数
        self.orbit_target = Vector3(0, 0, 0)
        self.orbit_distance = 5.0
        self.orbit_yaw = 0.0
        self.orbit_pitch = 0.0
        
        # 第一人称相机参数
        self.first_person_yaw = 0.0
        self.first_person_pitch = 0.0
        
        # 初始化相机参数
        self._initialize_camera_params()
        
    def set_ui_active(self, active):
        """设置UI是否激活
        
        Args:
            active: UI是否激活
        """
        self.is_ui_active = active
    
    def set_mode(self, mode):
        """设置相机模式
        
        Args:
            mode: 相机模式，取值为Mode.FREE, Mode.ORBIT, Mode.FIRST_PERSON
        """
        self.mode = mode
        self._update_camera_from_mode()
    
    def get_mode(self):
        """获取当前相机模式
        
        Returns:
            int: 当前相机模式
        """
        return self.mode
    
    def toggle_mode(self):
        """切换相机模式"""
        self.mode = (self.mode + 1) % 3
        self._update_camera_from_mode()
    
    def set_orbit_target(self, target):
        """设置轨道相机的目标点
        
        Args:
            target: 目标点坐标
        """
        self.orbit_target = target
        if self.mode == self.Mode.ORBIT:
            self._update_orbit_camera()
    
    def _initialize_camera_params(self):
        """初始化相机参数
        
        从当前相机位置计算初始的轨道相机和第一人称相机参数
        """
        # 初始化轨道相机参数
        camera_pos = self.camera.get_position()
        self.orbit_distance = (camera_pos - self.orbit_target).length()
        
        # 计算初始的yaw和pitch
        delta = camera_pos - self.orbit_target
        self.orbit_yaw = -math.atan2(delta.z, delta.x)
        self.orbit_pitch = math.atan2(delta.y, math.sqrt(delta.x * delta.x + delta.z * delta.z))
        
        # 初始化第一人称相机参数
        self.first_person_yaw = self.orbit_yaw
        self.first_person_pitch = self.orbit_pitch
    
    def handle_mouse_down(self, button, x, y):
        """处理鼠标按下事件
        
        Args:
            button: 鼠标按钮（1=左键，2=中键，3=右键）
            x: 鼠标X坐标
            y: 鼠标Y坐标
            
        Returns:
            bool: 是否处理了事件
        """
        if self.is_ui_active:
            return False
        
        if button == self.MOUSE_LEFT:  # 左键
            self.is_mouse_down[0] = True
        elif button == self.MOUSE_MIDDLE:  # 中键
            self.is_mouse_down[1] = True
        elif button == self.MOUSE_RIGHT:  # 右键
            self.is_mouse_down[2] = True
        
        self.last_mouse_pos = [x, y]
        return True
    
    def handle_mouse_up(self, button, x, y):
        """处理鼠标释放事件
        
        Args:
            button: 鼠标按钮（1=左键，2=中键，3=右键）
            x: 鼠标X坐标
            y: 鼠标Y坐标
            
        Returns:
            bool: 是否处理了事件
        """
        if button == self.MOUSE_LEFT:
            self.is_mouse_down[0] = False
        elif button == self.MOUSE_MIDDLE:
            self.is_mouse_down[1] = False
        elif button == self.MOUSE_RIGHT:
            self.is_mouse_down[2] = False
        
        self.last_mouse_pos = [x, y]
        return True
    
    def handle_mouse_move(self, x, y):
        """处理鼠标移动事件
        
        Args:
            x: 鼠标X坐标
            y: 鼠标Y坐标
            
        Returns:
            bool: 是否处理了事件
        """
        dx = x - self.last_mouse_pos[0]
        dy = y - self.last_mouse_pos[1]
        
        # 无论UI是否激活，都更新鼠标位置
        self.last_mouse_pos = [x, y]
        
        # UI激活时，不处理相机移动
        if self.is_ui_active:
            return False
        
        if self.mode == self.Mode.FREE:
            self._handle_free_camera_move(dx, dy)
        elif self.mode == self.Mode.ORBIT:
            self._handle_orbit_camera_move(dx, dy)
        elif self.mode == self.Mode.FIRST_PERSON:
            self._handle_first_person_camera_move(dx, dy)
        
        return True
    
    def _get_pan_speed(self):
        """获取平移速度
        
        Returns:
            float: 平移速度
        """
        return self.move_speed * 0.01
    
    def _clamp_pitch(self, pitch):
        """限制pitch范围，避免相机翻转
        
        Args:
            pitch: 原始pitch值
            
        Returns:
            float: 限制后的pitch值
        """
        return max(self.MIN_PITCH, min(self.MAX_PITCH, pitch))
    
    def _handle_free_camera_move(self, dx, dy):
        """处理自由相机移动"""
        if self.is_mouse_down[2]:  # 右键旋转
            self.camera.rotate_yaw(dx * self.rotate_speed)
            self.camera.rotate_pitch(-dy * self.rotate_speed)
        elif self.is_mouse_down[1]:  # 中键平移
            move_speed = self._get_pan_speed()
            self.camera.move_right(-dx * move_speed)
            self.camera.move_up(dy * move_speed)
    
    def _handle_orbit_camera_move(self, dx, dy):
        """处理轨道相机移动"""
        if self.is_mouse_down[2]:  # 右键旋转
            self.orbit_yaw += dx * self.rotate_speed
            self.orbit_pitch += dy * self.rotate_speed
            
            # 限制pitch范围，避免相机翻转
            self.orbit_pitch = self._clamp_pitch(self.orbit_pitch)
            
            # 更新轨道相机
            self._update_orbit_camera()
        elif self.is_mouse_down[1]:  # 中键平移
            move_speed = self._get_pan_speed()
            
            # 计算平移方向
            camera_right = self.camera.get_right()
            camera_up = self.camera.get_up()
            
            # 平移目标点
            self.orbit_target += camera_right * (-dx * move_speed) + camera_up * (dy * move_speed)
            
            # 更新轨道相机
            self._update_orbit_camera()
    
    def _handle_first_person_camera_move(self, dx, dy):
        """处理第一人称相机移动"""
        if self.is_mouse_down[2]:  # 右键旋转
            self.first_person_yaw += dx * self.rotate_speed
            self.first_person_pitch += dy * self.rotate_speed
            
            # 限制pitch范围，避免相机翻转
            self.first_person_pitch = self._clamp_pitch(self.first_person_pitch)
            
            # 更新第一人称相机
            self._update_first_person_camera()
    
    def handle_mouse_wheel(self, delta):
        """处理鼠标滚轮事件
        
        Args:
            delta: 滚轮滚动量（正数表示向前滚动，负数表示向后滚动）
            
        Returns:
            bool: 是否处理了事件
        """
        if self.is_ui_active:
            return False
        
        if self.mode == self.Mode.FREE:
            self.camera.move_forward(delta * self.zoom_speed)
        elif self.mode == self.Mode.ORBIT:
            # 轨道相机缩放
            self.orbit_distance -= delta * self.zoom_speed
            self.orbit_distance = max(self.MIN_ORBIT_DISTANCE, self.orbit_distance)  # 限制最小距离
            self._update_orbit_camera()
        elif self.mode == self.Mode.FIRST_PERSON:
            # 第一人称相机前进/后退
            self.camera.move_forward(delta * self.zoom_speed)
        
        return True
    
    def handle_key_down(self, key):
        # 处理全局键盘事件，即使UI激活
        if key == ord('c') or key == ord('C'):
            # 切换相机模式
            self.toggle_mode()
            return True
        
        # UI激活时，只处理全局事件，忽略相机移动事件
        if self.is_ui_active:
            return False
        
        # 使用ASCII码处理键盘事件，避免依赖pygame
        if key == ord('w') or key == ord('W'):
            self.camera.move_forward(self.move_speed)
        elif key == ord('s') or key == ord('S'):
            self.camera.move_forward(-self.move_speed)
        elif key == ord('a') or key == ord('A'):
            self.camera.move_right(-self.move_speed)
        elif key == ord('d') or key == ord('D'):
            self.camera.move_right(self.move_speed)
        elif key == ord('q') or key == ord('Q'):
            self.camera.move_up(-self.move_speed)
        elif key == ord('e') or key == ord('E'):
            self.camera.move_up(self.move_speed)
        
        return True
    
    def handle_key_up(self, key):
        # 键盘释放事件处理
        # 可以在这里添加按键释放时的逻辑，比如停止持续移动等
        return False
    
    def orbit(self, deltaX, deltaY):
        """HTML UI调用的轨道相机控制方法
        
        Args:
            deltaX: X方向的旋转增量
            deltaY: Y方向的旋转增量
        """
        # 如果当前不是轨道模式，切换到轨道模式
        if self.mode != self.Mode.ORBIT:
            self.set_mode(self.Mode.ORBIT)
        
        # 更新轨道相机参数
        self.orbit_yaw += deltaX * self.rotate_speed
        self.orbit_pitch -= deltaY * self.rotate_speed
        
        # 限制pitch范围，避免相机翻转
        self.orbit_pitch = self._clamp_pitch(self.orbit_pitch)
        
        # 更新轨道相机
        self._update_orbit_camera()
    
    def pan(self, deltaX, deltaY):
        """HTML UI调用的相机平移方法
        
        Args:
            deltaX: X方向的平移增量
            deltaY: Y方向的平移增量
        """
        if self.mode != self.Mode.ORBIT:
            self.set_mode(self.Mode.ORBIT)
        
        move_speed = self._get_pan_speed()
        
        # 计算平移方向
        camera_right = self.camera.get_right()
        camera_up = self.camera.get_up()
        
        # 平移目标点
        self.orbit_target += camera_right * (-deltaX * move_speed) + camera_up * (deltaY * move_speed)
        
        # 更新轨道相机
        self._update_orbit_camera()
    
    def zoom(self, delta):
        """HTML UI调用的相机缩放方法
        
        Args:
            delta: 缩放增量，正数表示放大，负数表示缩小
        """
        if self.mode == self.Mode.ORBIT:
            # 轨道相机缩放
            self.orbit_distance -= delta * self.zoom_speed
            self.orbit_distance = max(self.MIN_ORBIT_DISTANCE, self.orbit_distance)  # 限制最小距离
            self._update_orbit_camera()
        else:
            # 其他相机模式下的缩放
            self.camera.move_forward(delta * self.zoom_speed)
    
    def move(self, direction):
        """HTML UI调用的相机方向移动方法
        
        Args:
            direction: 移动方向，如 'forward', 'backward', 'left', 'right', 'up', 'down'
        """
        if direction == 'forward':
            self.camera.move_forward(self.move_speed)
        elif direction == 'backward':
            self.camera.move_forward(-self.move_speed)
        elif direction == 'left':
            self.camera.move_right(-self.move_speed)
        elif direction == 'right':
            self.camera.move_right(self.move_speed)
        elif direction == 'up':
            self.camera.move_up(self.move_speed)
        elif direction == 'down':
            self.camera.move_up(-self.move_speed)
    
    def set_top_view(self):
        """设置顶视图"""
        self.orbit_target = Vector3(0, 0, 0)
        self.orbit_yaw = 0
        self.orbit_pitch = math.pi/2
        self.orbit_distance = 10
        self.set_mode(self.Mode.ORBIT)
    
    def set_front_view(self):
        """设置前视图"""
        self.orbit_target = Vector3(0, 0, 0)
        self.orbit_yaw = math.pi
        self.orbit_pitch = 0
        self.orbit_distance = 10
        self.set_mode(self.Mode.ORBIT)
    
    def set_right_view(self):
        """设置右视图"""
        self.orbit_target = Vector3(0, 0, 0)
        self.orbit_yaw = math.pi/2
        self.orbit_pitch = 0
        self.orbit_distance = 10
        self.set_mode(self.Mode.ORBIT)
    
    def _update_camera_from_mode(self):
        """根据当前模式更新相机"""
        if self.mode == self.Mode.ORBIT:
            self._update_orbit_camera()
        elif self.mode == self.Mode.FIRST_PERSON:
            self._update_first_person_camera()
    
    def _update_orbit_camera(self):
        """更新轨道相机"""
        # 计算相机位置
        x = self.orbit_target.x + self.orbit_distance * math.cos(self.orbit_pitch) * math.cos(self.orbit_yaw)
        y = self.orbit_target.y + self.orbit_distance * math.sin(self.orbit_pitch)
        z = self.orbit_target.z + self.orbit_distance * math.cos(self.orbit_pitch) * math.sin(self.orbit_yaw)
        
        # 设置相机位置
        self.camera.set_position(Vector3(x, y, z))
        
        # 看向目标点
        self.camera.look_at(self.orbit_target)
    
    def _update_first_person_camera(self):
        """更新第一人称相机"""
        # 创建旋转四元数
        rotation = Quaternion.from_euler(self.first_person_yaw, self.first_person_pitch, 0)
        
        # 设置相机旋转
        self.camera.set_rotation(rotation)
    
    def update(self, delta_time):
        """更新相机控制器
        
        Args:
            delta_time: 帧间隔时间（秒）
        """
        # 平滑过渡逻辑
        if hasattr(self, '_smooth_transition') and self._smooth_transition:
            self._update_smooth_transition(delta_time)
        
        # 自动旋转逻辑
        if hasattr(self, '_auto_rotate') and self._auto_rotate:
            self._update_auto_rotate(delta_time)
        
        # 平滑跟随目标
        if hasattr(self, '_smooth_follow') and self._smooth_follow:
            self._update_smooth_follow(delta_time)
    
    def _update_smooth_transition(self, delta_time):
        """更新平滑过渡"""
        # 实现平滑过渡逻辑
        pass
    
    def _update_auto_rotate(self, delta_time):
        """更新自动旋转"""
        # 自动旋转相机
        if self.mode == self.Mode.ORBIT:
            self.orbit_yaw += 0.5 * delta_time
            self._update_orbit_camera()
    
    def _update_smooth_follow(self, delta_time):
        """更新平滑跟随"""
        # 实现平滑跟随逻辑
        pass
    
    def set_auto_rotate(self, enable):
        """设置是否启用自动旋转
        
        Args:
            enable: 是否启用自动旋转
        """
        self._auto_rotate = enable
    
    def set_smooth_follow(self, enable):
        """设置是否启用平滑跟随
        
        Args:
            enable: 是否启用平滑跟随
        """
        self._smooth_follow = enable
