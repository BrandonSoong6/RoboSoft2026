# --------------------------------------------------------------------
        # 【修改 2: 核心路径采样、误差计算和 Quiver 更新】
        # --------------------------------------------------------------------
        
        # 1. 初始化误差向量数据
        robot_sampled_coords = np.array([]).reshape(0, 2)
        target_sampled_coords = np.array([]).reshape(0, 2)
        tracking_error_vectors = np.array([]).reshape(0, 2)
        
        # 优化判断逻辑，并使用内嵌 Try-Except 来捕获 Quiver 错误
        if target_u_sampled is not None and tck_robot is not None and target_path_tck is not None:
            try:
                # 获取机器人路径采样点 (橙色方块)
                robot_sampled_coords = reparameterize_robot_path(
                    u_sampled_target=target_u_sampled, 
                    tck_robot=tck_robot,
                    tck_target=target_path_tck
                )
                
                # 获取目标路径采样点 (红色圆圈)
                sampled_coords_target = splev(target_u_sampled, target_path_tck, der=0)
                target_sampled_coords = np.vstack(sampled_coords_target).T
                
                # 只有当两个采样点数组长度一致时，才计算误差和更新绘图
                if robot_sampled_coords.shape[0] == target_sampled_coords.shape[0] and robot_sampled_coords.shape[0] > 0:
                    
                    # 计算误差向量: 目标 - 机器人 (Target - Robot)
                    tracking_error_vectors = target_sampled_coords - robot_sampled_coords
                    
                    # 更新 Quiver 图 (误差向量)
                    X_start = robot_sampled_coords[:, 0]
                    Y_start = robot_sampled_coords[:, 1]
                    U_vec = tracking_error_vectors[:, 0]
                    V_vec = tracking_error_vectors[:, 1]
                    
                    mapping_quiver.set_offsets(np.vstack([X_start, Y_start]).T)
                    mapping_quiver.set_UVC(U_vec, V_vec)
                    
                    # 更新采样点 scatter
                    robot_sampled_scatter.set_offsets(robot_sampled_coords)
                    target_adaptive_scatter.set_offsets(target_sampled_coords)
                    
                else:
                    # 数据长度不匹配或采样点为空，清空绘图
                    raise ValueError("Sampled path arrays are empty or length mismatch.")

            except Exception as e:
                # 采样失败，清空所有相关点和向量
                # print(f"Applying density distribution or error calculation failed: {e}")
                robot_sampled_scatter.set_offsets(np.array([]).reshape(0, 2))
                target_adaptive_scatter.set_offsets(np.array([]).reshape(0, 2))
                mapping_quiver.set_offsets(np.array([]).reshape(0, 2))
                mapping_quiver.set_UVC([], [])
        else:
            # 如果目标路径或机器人路径不存在，清空点和向量
            robot_sampled_scatter.set_offsets(np.array([]).reshape(0, 2))
            target_adaptive_scatter.set_offsets(np.array([]).reshape(0, 2))
            mapping_quiver.set_offsets(np.array([]).reshape(0, 2))
            mapping_quiver.set_UVC([], [])