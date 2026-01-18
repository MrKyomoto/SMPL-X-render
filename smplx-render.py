# main.py
#!/usr/bin/env python3
"""
SMPL-X 3D人体动画控制系统 - 程序入口
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont

# 导入主界面
from ui import HumanAnimationSystem

def main():
    """程序入口函数"""
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        window = HumanAnimationSystem()
        window.show()
        print("=" * 70)
        print("SMPL-X 3D人体动画控制系统")
        print("=" * 70)
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        print(f"程序错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
