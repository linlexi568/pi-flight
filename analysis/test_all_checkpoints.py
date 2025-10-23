"""
测试所有checkpoint文件的性能
使用test_extreme预设在stress扰动条件下评估
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "01_pi_flight" / "results" / "checkpoints"
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
VERIFY_SCRIPT = ROOT_DIR / "utilities" / "verify_program.py"


def get_all_checkpoints() -> List[Path]:
    """获取所有checkpoint文件"""
    if not CHECKPOINT_DIR.exists():
        print(f"错误: checkpoint目录不存在: {CHECKPOINT_DIR}")
        return []
    
    checkpoints = sorted(CHECKPOINT_DIR.glob("best_program_iter_*.json"))
    return checkpoints


def extract_iteration(checkpoint_path: Path) -> int:
    """从文件名提取迭代次数"""
    name = checkpoint_path.stem  # 如 "best_program_iter_002800"
    iter_str = name.split("_")[-1]  # "002800"
    return int(iter_str)


def read_checkpoint_info(checkpoint_path: Path) -> Dict:
    """读取checkpoint的基本信息"""
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        num_rules = len(data.get('rules', []))
        verified_score = data.get('verified_score', None)
        train_score = data.get('train_score', None)
        
        return {
            'path': checkpoint_path,
            'iteration': extract_iteration(checkpoint_path),
            'num_rules': num_rules,
            'verified_score': verified_score,
            'train_score': train_score,
        }
    except Exception as e:
        print(f"读取 {checkpoint_path.name} 失败: {e}")
        return None


def test_checkpoint(checkpoint_path: Path, inplace: bool = True) -> Tuple[float, Dict]:
    """
    测试单个checkpoint
    
    Args:
        checkpoint_path: checkpoint文件路径
        inplace: 是否写回verified_score到文件
    
    Returns:
        (score, per_traj_dict)
    """
    cmd = [
        str(VENV_PYTHON),
        str(VERIFY_SCRIPT),
        "--program", str(checkpoint_path),
        "--traj_preset", "test_extreme",
        "--aggregate", "harmonic",
        "--disturbance", "stress",
        "--duration", "25",
        "--log-skip", "2",
        "--clip-D", "1.2",
        "--compose-by-gain",
    ]
    
    if inplace:
        cmd.append("--inplace")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT_DIR)
        )
        
        # 解析输出找到得分
        output = result.stdout
        score = None
        per_traj = {}
        
        for line in output.split('\n'):
            if '[Verified]' in line and '聚合得分:' in line:
                # 提取得分
                parts = line.split('聚合得分:')
                if len(parts) > 1:
                    score_str = parts[1].strip().split()[0]
                    score = float(score_str)
            
            if '逐轨迹:' in line:
                # 尝试解析逐轨迹得分
                try:
                    import re
                    # 查找列表部分
                    match = re.search(r'\[({.*})\]', line.replace('\n', ''))
                    if match:
                        import ast
                        traj_list = ast.literal_eval('[' + match.group(1) + ']')
                        for item in traj_list:
                            per_traj[item['traj']] = item['reward']
                except:
                    pass
        
        if score is None:
            print(f"警告: 无法解析 {checkpoint_path.name} 的得分")
            print(f"输出: {output[:500]}")
        
        return score, per_traj
    
    except Exception as e:
        print(f"测试 {checkpoint_path.name} 失败: {e}")
        return None, {}


def test_all_checkpoints(force_retest: bool = False, max_count: int = None):
    """
    测试所有checkpoint
    
    Args:
        force_retest: 强制重新测试（即使已有verified_score）
        max_count: 最多测试多少个（None=全部）
    """
    checkpoints = get_all_checkpoints()
    
    if not checkpoints:
        print("没有找到checkpoint文件")
        return
    
    print(f"找到 {len(checkpoints)} 个checkpoint文件")
    print("=" * 80)
    
    results = []
    
    for idx, ckpt_path in enumerate(checkpoints):
        if max_count and idx >= max_count:
            break
        
        # 读取基本信息
        info = read_checkpoint_info(ckpt_path)
        if not info:
            continue
        
        print(f"\n[{idx+1}/{len(checkpoints)}] 测试 {ckpt_path.name}")
        print(f"  迭代: {info['iteration']}, 规则数: {info['num_rules']}")
        
        # 检查是否已有verified_score
        if not force_retest and info['verified_score'] is not None:
            print(f"  已有verified_score: {info['verified_score']:.6f} (跳过)")
            results.append({
                'iteration': info['iteration'],
                'num_rules': info['num_rules'],
                'verified_score': info['verified_score'],
                'train_score': info['train_score'],
                'status': 'cached'
            })
            continue
        
        # 执行测试
        score, per_traj = test_checkpoint(ckpt_path, inplace=True)
        
        if score is not None:
            print(f"  ✅ 验证得分: {score:.6f}")
            results.append({
                'iteration': info['iteration'],
                'num_rules': info['num_rules'],
                'verified_score': score,
                'train_score': info['train_score'],
                'per_traj': per_traj,
                'status': 'tested'
            })
        else:
            print(f"  ❌ 测试失败")
            results.append({
                'iteration': info['iteration'],
                'num_rules': info['num_rules'],
                'verified_score': None,
                'train_score': info['train_score'],
                'status': 'failed'
            })
    
    # 保存结果
    save_results(results)
    
    # 显示摘要
    print("\n" + "=" * 80)
    print("测试完成摘要")
    print("=" * 80)
    display_summary(results)


def save_results(results: List[Dict]):
    """保存测试结果到CSV"""
    if not results:
        return
    
    df = pd.DataFrame(results)
    output_file = ROOT_DIR / "analysis" / "checkpoint_test_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")


def display_summary(results: List[Dict]):
    """显示测试结果摘要"""
    df = pd.DataFrame(results)
    
    print(f"\n总计checkpoint数: {len(df)}")
    print(f"成功测试: {len(df[df['verified_score'].notna()])}")
    print(f"使用缓存: {len(df[df['status'] == 'cached'])}")
    print(f"测试失败: {len(df[df['status'] == 'failed'])}")
    
    if len(df[df['verified_score'].notna()]) > 0:
        best_row = df.loc[df['verified_score'].idxmax()]
        print(f"\n最佳checkpoint:")
        print(f"  迭代: {best_row['iteration']}")
        print(f"  得分: {best_row['verified_score']:.6f}")
        print(f"  规则数: {best_row['num_rules']}")
        
        print(f"\n得分统计:")
        print(df['verified_score'].describe())


def main():
    import argparse
    parser = argparse.ArgumentParser(description="测试所有checkpoint性能")
    parser.add_argument('--force', action='store_true', help='强制重新测试（忽略已有verified_score）')
    parser.add_argument('--max', type=int, default=None, help='最多测试多少个checkpoint')
    args = parser.parse_args()
    
    test_all_checkpoints(force_retest=args.force, max_count=args.max)


if __name__ == "__main__":
    main()
