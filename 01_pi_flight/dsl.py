# Copied from 01_pi_light/dsl.py for package rename
import numpy as np, math, abc
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
        if self.op=='abs': return abs(v)
        if self.op=='sign': return np.sign(v)
        if self.op=='sin': return math.sin(v)
        if self.op=='cos': return math.cos(v)
        if self.op=='tan':
            # 安全裁剪：避免 tan 在奇异点爆炸
            try:
                val=math.tan(v)
            except Exception:
                val=0.0
            if val>10: val=10
            elif val<-10: val=-10
            return val
        if self.op=='log1p':
            # log1p(|v|) 压缩幅度；避免 log(<=0) 问题
            try:
                return math.log1p(abs(v))
            except Exception:
                return 0.0
        if self.op=='sqrt':
            try:
                return math.sqrt(abs(v))
            except Exception:
                return 0.0
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
