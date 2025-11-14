# Copied from 01_pi_light/dsl.py for package rename
import numpy as np, math, abc
from collections import deque
class ProgramNode(abc.ABC):
    def evaluate(self, state_dict: dict) -> float: ...
    def __str__(self): return 'ProgramNode'
class TerminalNode(ProgramNode):
    def __init__(self,value): self.value=value
    def evaluate(self,state_dict):
        if isinstance(self.value,str): return state_dict.get(self.value,0.0)
        return self.value
    def __str__(self):
        if isinstance(self.value,float): return f"{self.value:.2f}"; return str(self.value)
class UnaryOpNode(ProgramNode):
    def __init__(self,op,child): self.op=op; self.child=child
    def evaluate(self,sd):
        v=self.child.evaluate(sd)
        op = self.op
        # 基础一元操作
        if op=='abs': return abs(v)
        if op=='sign': return np.sign(v)
        if op=='sin': return math.sin(v)
        if op=='cos': return math.cos(v)
        if op=='tan':
            # 安全裁剪：避免 tan 在奇异点爆炸
            try:
                val=math.tan(v)
            except Exception:
                val=0.0
            if val>10: val=10
            elif val<-10: val=-10
            return val
        if op=='log1p':
            # log1p(|v|) 压缩幅度；避免 log(<=0) 问题
            try:
                return math.log1p(abs(v))
            except Exception:
                return 0.0
        if op=='sqrt':
            try:
                return math.sqrt(abs(v))
            except Exception:
                return 0.0

        # 时序/稳定性原语（参数编码在 op 字符串中）
        try:
            # 统一解析前缀和参数，以":"分割
            prefix, *args = op.split(':')
        except Exception:
            prefix, args = op, []

        # ema:alpha  → y_t = (1-α) y_{t-1} + α v
        if prefix=='ema':
            alpha = float(args[0]) if args else 0.2
            if not hasattr(self, '_ema_prev'):
                self._ema_prev = 0.0
            y = (1.0 - alpha) * self._ema_prev + alpha * v
            self._ema_prev = y
            return y

        # delay:k  → 返回 k 步前的 v
        if prefix=='delay':
            k = int(float(args[0])) if args else 1
            if k < 1: k = 1
            if not hasattr(self, '_buf'):
                self._buf = deque(maxlen=k)
            # 若 maxlen 变化，重建缓冲
            if isinstance(self._buf, deque) and self._buf.maxlen != k:
                self._buf = deque(list(self._buf), maxlen=k)
            # 读取输出：不足 k 步时返回 0
            out = self._buf[0] if len(self._buf) == k else 0.0
            self._buf.appendleft(v)
            return float(out)

        # diff:k  → v - delay(v,k)
        if prefix=='diff':
            k = int(float(args[0])) if args else 1
            if k < 1: k = 1
            if not hasattr(self, '_buf_d'):
                self._buf_d = deque(maxlen=k)
            if isinstance(self._buf_d, deque) and self._buf_d.maxlen != k:
                self._buf_d = deque(list(self._buf_d), maxlen=k)
            prev = self._buf_d[0] if len(self._buf_d) == k else v
            self._buf_d.appendleft(v)
            return float(v - prev)

        # clamp:lo:hi  → 限幅
        if prefix=='clamp':
            lo = float(args[0]) if len(args)>=1 else -5.0
            hi = float(args[1]) if len(args)>=2 else 5.0
            if lo>hi: lo,hi = hi,lo
            return float(min(max(v, lo), hi))

        # deadzone:eps  → 小误差置零
        if prefix=='deadzone':
            eps = float(args[0]) if args else 0.01
            if abs(v) <= eps: return 0.0
            return float(v - math.copysign(eps, v))

        # rate: r  → 斜率限幅（Δt=1）
        if prefix=='rate' or prefix=='rate_limit':
            r = float(args[0]) if args else 1.0
            if not hasattr(self, '_rate_prev'):
                self._rate_prev = 0.0
            y_prev = self._rate_prev
            lo = y_prev - r
            hi = y_prev + r
            y = min(max(v, lo), hi)
            self._rate_prev = y
            return float(y)

        # smooth:s  → 平滑限幅（tanh 型）
        if prefix=='smooth' or prefix=='smoothstep':
            s = float(args[0]) if args else 1.0
            s = max(1e-6, s)
            return float(s * math.tanh(v / s))

        raise ValueError('未知的一元操作:'+self.op)
    def __str__(self): return f"{self.op}({self.child})"
class BinaryOpNode(ProgramNode):
    def __init__(self,op,left,right): self.op=op; self.left=left; self.right=right
    def evaluate(self,sd):
        l=self.left.evaluate(sd); r=self.right.evaluate(sd)
        if self.op=='+': return l+r
        if self.op=='-': return l-r
        if self.op=='/':
            try:
                return l/ (r if abs(r) > 1e-12 else (1e-12 if r>=0 else -1e-12))
            except Exception:
                return 0.0
        if self.op=='>': return 1.0 if l>r else 0.0
        if self.op=='<': return 1.0 if l<r else 0.0
        if self.op=='max': return max(l,r)
        if self.op=='min': return min(l,r)
        if self.op=='*': return l*r
        if self.op=='==': return 1.0 if l==r else 0.0
        if self.op=='!=': return 1.0 if l!=r else 0.0
        raise ValueError('未知的二元操作:'+self.op)
    def __str__(self): return f"({self.left} {self.op} {self.right})"
class IfNode(ProgramNode):
    def __init__(self,condition,then_branch,else_branch): self.condition=condition; self.then_branch=then_branch; self.else_branch=else_branch
    def evaluate(self,sd): return self.then_branch.evaluate(sd) if self.condition.evaluate(sd)>0 else self.else_branch.evaluate(sd)
    def __str__(self): return f"if {self.condition} then ({self.then_branch}) else ({self.else_branch})"
