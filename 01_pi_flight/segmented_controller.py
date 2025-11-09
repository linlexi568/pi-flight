# Copied from 01_pi_light/segmented_controller.py for package rename
from typing import Dict, Any, Callable, List, Union, Tuple
import numpy as np
from scipy.spatial.transform import Rotation
from .local_pid import SimplePIDControl as DSLPIDControl
from .local_pid import LocalDroneModel as DroneModel
from .dsl import ProgramNode, UnaryOpNode, BinaryOpNode, TerminalNode, IfNode

class PiLightSegmentedPIDController(DSLPIDControl):
    def __init__(self,
                 drone_model: Union[DroneModel, str] = "cf2x",
                 program: List[Dict[str, Any]] = None,
                 suppress_init_print: bool=False,
                 compose_by_gain: bool=False,
                 clip_P: Union[float, None]=None,
                 clip_I: Union[float, None]=None,
                 clip_D: Union[float, None]=None,
                 semantics: Union[str, None] = None,
                 require_k: int = 0,
                 blend_topk_k: int = 2,
                 gain_slew_limit: Union[float, List[float], Tuple[float, ...], None] = None,
                 min_hold_steps: int = 0):
        super().__init__(drone_model=drone_model)
        self.segments=[]; self._segment_hit_counter={}
        self.default_gains={'P':np.copy(self.P_COEFF_TOR),'I':np.copy(self.I_COEFF_TOR),'D':np.copy(self.D_COEFF_TOR)}
        self.gain_history=[]; self.last_rule_name=None
        self._suppress_init_print = suppress_init_print
        if semantics is None:
            self._semantics = 'compose_by_gain' if bool(compose_by_gain) else 'first_match'
        else:
            s = str(semantics).strip().lower()
            self._semantics = s if s in ('first_match','compose_by_gain','blend_topk') else ('compose_by_gain' if bool(compose_by_gain) else 'first_match')
        self._compose_by_gain = (self._semantics == 'compose_by_gain')
        try:
            self._require_k = max(0, int(require_k))
        except Exception:
            self._require_k = 0
        try:
            self._blend_topk_k = max(1, int(blend_topk_k))
        except Exception:
            self._blend_topk_k = 2
        def _mk_clip(v):
            try:
                if v is None: return None
                v=float(v); return None if v<=0 else v
            except Exception:
                return None
        self._clip_max={'P': _mk_clip(clip_P), 'I': _mk_clip(clip_I), 'D': _mk_clip(clip_D)}
        self._metric_steps = 0
        self._metric_overlap_sum = 0.0
        self._metric_action_diff_step_sum = 0.0
        self._slew_active = False
        self._slew = np.array([0.0, 0.0, 0.0], dtype=float)
        if gain_slew_limit is not None:
            try:
                v_arr = None
                if isinstance(gain_slew_limit, (int, float)):
                    val = float(gain_slew_limit)
                    v_arr = np.array([val, val, val], dtype=float)
                elif isinstance(gain_slew_limit, (list, tuple)):
                    seq = [float(x) for x in gain_slew_limit]
                    if len(seq) == 1:
                        seq = [seq[0], seq[0], seq[0]]
                    elif len(seq) == 2:
                        seq = [seq[0], seq[1], seq[1]]
                    else:
                        seq = [seq[0], seq[1], seq[2]]
                    v_arr = np.array(seq, dtype=float)
                elif isinstance(gain_slew_limit, str):
                    parts = [p.strip() for p in gain_slew_limit.split(',') if p.strip()]
                    if len(parts) == 0:
                        v_arr = None
                    else:
                        nums = [float(parts[0])]
                        if len(parts) >= 2:
                            nums.append(float(parts[1]))
                        else:
                            nums.append(nums[0])
                        if len(parts) >= 3:
                            nums.append(float(parts[2]))
                        else:
                            nums.append(nums[-1])
                        v_arr = np.array(nums, dtype=float)
                if v_arr is not None:
                    v_arr = np.maximum(0.0, v_arr)
                    if np.any(v_arr > 0):
                        self._slew_active = True
                        self._slew = v_arr
            except Exception:
                pass
        self._prev_mult = np.array([1.0, 1.0, 1.0], dtype=float)
        try:
            self._min_hold = max(0, int(min_hold_steps))
        except Exception:
            self._min_hold = 0
        self._hold_rem = 0
        self._last_planned_name = None
        self._last_planned_mult = np.array([1.0, 1.0, 1.0], dtype=float)
        if program:
            self._parse_program_to_segments(program)
    def _parse_program_to_segments(self, program: List[Dict[str, Any]]):
        if not isinstance(program,list): return
        for i,rule_node in enumerate(program):
            if isinstance(rule_node,dict) and 'condition' in rule_node and 'action' in rule_node:
                cond_ast=rule_node['condition']; action_ast=rule_node['action']
                cond_fun=self._ast_to_lambda(cond_ast); mults=self._extract_gains_from_action(action_ast)
                if cond_fun and mults:
                    seg={'name':f"Rule_{i}", 'condition':cond_fun, 'multipliers':mults, 'condition_ast': cond_ast}
                    self.segments.append(seg); self._segment_hit_counter[seg['name']]=0
                    if not self._suppress_init_print:
                        print(f"[控制器初始化] 添加分段规则 '{seg['name']}'")
    def _ast_to_lambda(self,node:ProgramNode)->Callable[[Dict[str,float]],Any]:
        if isinstance(node,BinaryOpNode):
            l=self._ast_to_lambda(node.left); r=self._ast_to_lambda(node.right); op=node.op
            if op=='+': return lambda s: l(s)+r(s)
            if op=='-': return lambda s: l(s)-r(s)
            if op=='*': return lambda s: l(s)*r(s)
            if op=='>': return lambda s: 1.0 if l(s)>r(s) else 0.0
            if op=='<': return lambda s: 1.0 if l(s)<r(s) else 0.0
            if op=='==': return lambda s: 1.0 if l(s)==r(s) else 0.0
            if op!='!=' and op not in ('max','min'):
                pass
            if op=='!=': return lambda s: 1.0 if l(s)!=r(s) else 0.0
            if op=='max': return lambda s: max(l(s),r(s))
            if op=='min': return lambda s: min(l(s),r(s))
            return lambda s:0.0
        if isinstance(node,UnaryOpNode):
            ch=self._ast_to_lambda(node.child); op=node.op
            if op=='abs': return lambda s: abs(ch(s))
            if op=='sin': return lambda s: float(np.sin(ch(s)))
            if op=='cos': return lambda s: float(np.cos(ch(s)))
            if op=='sign': return lambda s: float(np.sign(ch(s)))
            if op=='tan':
                return lambda s: float(np.clip(np.tan(ch(s)), -10.0, 10.0))
            if op=='log1p':
                return lambda s: float(np.log1p(abs(ch(s))))
            if op=='sqrt':
                return lambda s: float(np.sqrt(abs(ch(s))))
        if isinstance(node,TerminalNode):
            if isinstance(node.value,str): return lambda s: s.get(node.value,0.0)
            return lambda s: node.value
        if isinstance(node,IfNode):
            c=self._ast_to_lambda(node.condition); t=self._ast_to_lambda(node.then_branch); e=self._ast_to_lambda(node.else_branch)
            return lambda s: t(s) if c(s)>0 else e(s)
        return lambda s: False
    def _extract_gains_from_action(self,node:Any)->Dict[str,float]:
        mults={}
        if isinstance(node,list):
            for n in node:
                if isinstance(n,BinaryOpNode) and n.op=='set' and isinstance(n.left,TerminalNode) and isinstance(n.right,TerminalNode):
                    if n.left.value in ['P','I','D']:
                        mults[n.left.value]=n.right.value
        return mults
    def _cond_specificity(self, cond) -> float:
        from .dsl import BinaryOpNode, TerminalNode, UnaryOpNode
        def _base_var_name(node):
            cur=node
            while isinstance(cur, UnaryOpNode):
                cur=cur.child
            if isinstance(cur, TerminalNode) and isinstance(cur.value, str):
                return cur.value
            return None
        if not (hasattr(cond,'op') and hasattr(cond,'left') and hasattr(cond,'right')):
            return 0.0
        if not isinstance(cond, BinaryOpNode) or cond.op not in ('<','>'):
            return 0.0
        if not (isinstance(cond.right, TerminalNode) and isinstance(cond.right.value,(int,float))):
            return 0.0
        T=float(cond.right.value); var=_base_var_name(cond.left)
        caps={'lt_max':1.5,'gt_min':0.2}
        strict_caps={
            'pos_err_z': {'lt_max': 1.0, 'gt_min': 0.15},
            'pos_err_xy': {'lt_max': 1.6, 'gt_min': 0.2},
            'ang_vel_x': {'lt_max': 2.2, 'gt_min': 0.3},
            'ang_vel_y': {'lt_max': 2.2, 'gt_min': 0.3},
            'ang_vel_mag': {'lt_max': 2.5, 'gt_min': 0.4},
            'rpy_err_mag': {'lt_max': 1.8, 'gt_min': 0.2},
            'pos_err_z_abs': {'lt_max': 1.2, 'gt_min': 0.15},
            'err_i_z': {'lt_max': 2.0, 'gt_min': 0.25},
            'err_i_x': {'lt_max': 2.0, 'gt_min': 0.25},
            'err_i_y': {'lt_max': 2.0, 'gt_min': 0.25},
        }
        if isinstance(var,str) and var in strict_caps:
            caps=strict_caps[var]
        score=0.0
        if cond.op=='<':
            T_eff=max(0.0,T); ratio=min(1.0, T_eff/max(1e-6,caps['lt_max']))
            score=1.0 - ratio
        else:
            if T<=0:
                score=0.0
            else:
                score=min(1.0, T/max(1e-6, 3.0*caps['gt_min']))
        return float(max(0.0, min(1.0, score)))

    def _update_gains(self, state:Dict[str,float]):
        self.P_COEFF_TOR=np.copy(self.default_gains['P']); self.I_COEFF_TOR=np.copy(self.default_gains['I']); self.D_COEFF_TOR=np.copy(self.default_gains['D']); self.last_rule_name=None
        matches = []
        for idx, seg in enumerate(self.segments):
            try:
                if seg['condition'](state):
                    spec=self._cond_specificity(getattr(seg,'_raw_condition', None) or seg.get('raw_condition', None) or getattr(seg, 'condition_ast', None))
                    if not isinstance(spec, (int,float)) or spec < 0:
                        spec = 0.5
                    matches.append((idx, seg, float(spec), seg['multipliers']))
            except Exception:
                continue
        try:
            self._metric_steps += 1
            overlap_count = max(0, len(matches) - 1)
            self._metric_overlap_sum += float(overlap_count)
            if len(matches) > 1:
                vecs = []
                for _, _, _, mults in matches:
                    p = float(mults.get('P', 1.0)); i = float(mults.get('I', 1.0)); d = float(mults.get('D', 1.0))
                    vecs.append(np.array([p,i,d], dtype=float))
                diffs=[]
                for a in range(len(vecs)):
                    for b in range(a+1, len(vecs)):
                        diffs.append(float(np.linalg.norm(vecs[a]-vecs[b])))
                if diffs:
                    self._metric_action_diff_step_sum += float(np.mean(diffs))
        except Exception:
            pass

        sem = getattr(self, '_semantics', 'first_match')
        if sem == 'first_match':
            for idx, seg, _spec, _m in matches:
                m=seg['multipliers']
                if 'P' in m: self.P_COEFF_TOR*=m['P']
                if 'I' in m: self.I_COEFF_TOR*=m['I']
                if 'D' in m: self.D_COEFF_TOR*=m['D']
                self._segment_hit_counter[seg['name']]+=1; self.last_rule_name=seg['name']; break
        elif sem == 'compose_by_gain':
            chosen={'P': (None, -1.0), 'I': (None, -1.0), 'D': (None, -1.0)}
            last_hit=None
            for _idx, seg, spec, m in matches:
                for g in ('P','I','D'):
                    if g in m and spec > chosen[g][1]:
                        chosen[g]=(m[g], spec); last_hit=seg['name']
                self._segment_hit_counter[seg['name']]+=1
            if chosen['P'][0] is not None: self.P_COEFF_TOR*=chosen['P'][0]
            if chosen['I'][0] is not None: self.I_COEFF_TOR*=chosen['I'][0]
            if chosen['D'][0] is not None: self.D_COEFF_TOR*=chosen['D'][0]
            self.last_rule_name = last_hit
        elif sem == 'blend_topk':
            k = int(getattr(self, '_blend_topk_k', 2) or 2)
            req = int(getattr(self, '_require_k', 0) or 0)
            last_hit=None
            for g, arr in [('P', self.P_COEFF_TOR), ('I', self.I_COEFF_TOR), ('D', self.D_COEFF_TOR)]:
                cands = []
                for _idx, seg, spec, m in matches:
                    if g in m:
                        try:
                            cands.append((float(spec), float(m[g]), seg['name']))
                        except Exception:
                            cands.append((float(spec), m[g], seg['name']))
                if not cands:
                    continue
                cands.sort(key=lambda x: x[0], reverse=True)
                if req > 0 and len(cands) < req:
                    best = cands[0]
                    arr *= best[1]
                    last_hit = best[2]
                else:
                    top = cands[:max(1, k)]
                    weights = np.array([max(0.0, s) for (s, _m, _n) in top], dtype=float)
                    if float(np.sum(weights)) <= 1e-9:
                        weights = np.ones(len(top), dtype=float)
                    weights = weights / float(np.sum(weights))
                    mults = np.array([float(m) for (_s, m, _n) in top], dtype=float)
                    blended = float(np.sum(weights * mults))
                    arr *= blended
                    try:
                        argmax_id = int(np.argmax(weights))
                        last_hit = top[argmax_id][2]
                    except Exception:
                        last_hit = top[0][2]
            self.last_rule_name = last_hit
        else:
            for idx, seg, _spec, _m in matches:
                m=seg['multipliers']
                if 'P' in m: self.P_COEFF_TOR*=m['P']
                if 'I' in m: self.I_COEFF_TOR*=m['I']
                if 'D' in m: self.D_COEFF_TOR*=m['D']
                self._segment_hit_counter[seg['name']]+=1; self.last_rule_name=seg['name']; break
        def _avg(x):
            try: return float(np.mean(x))
            except: return float(x)
        def _safe_ratio(n, d):
            dn = float(_avg(d))
            nn = float(_avg(n))
            return float(nn / (dn + 1e-9))
        planned_mult = np.array([
            _safe_ratio(self.P_COEFF_TOR, self.default_gains['P']),
            _safe_ratio(self.I_COEFF_TOR, self.default_gains['I']),
            _safe_ratio(self.D_COEFF_TOR, self.default_gains['D'])
        ], dtype=float)
        planned_name = self.last_rule_name
        if getattr(self, '_min_hold', 0) > 0:
            if self._hold_rem > 0:
                planned_mult = np.array(self._last_planned_mult, dtype=float)
                planned_name = self._last_planned_name
                self._hold_rem = max(0, int(self._hold_rem) - 1)
            else:
                if (self._last_planned_name is not None) and (planned_name is not None) and (planned_name != self._last_planned_name):
                    planned_mult = np.array(self._last_planned_mult, dtype=float)
                    planned_name = self._last_planned_name
                    self._hold_rem = int(getattr(self, '_min_hold', 0))
                else:
                    self._last_planned_mult = np.array(planned_mult, dtype=float)
                    self._last_planned_name = planned_name
        else:
            self._last_planned_mult = np.array(planned_mult, dtype=float)
            self._last_planned_name = planned_name
        self.P_COEFF_TOR = self.default_gains['P'] * planned_mult[0]
        self.I_COEFF_TOR = self.default_gains['I'] * planned_mult[1]
        self.D_COEFF_TOR = self.default_gains['D'] * planned_mult[2]
        def _clip_inplace():
            for k, arr in [('P', self.P_COEFF_TOR), ('I', self.I_COEFF_TOR), ('D', self.D_COEFF_TOR)]:
                cmax=self._clip_max.get(k)
                if cmax is not None and cmax>0:
                    base=self.default_gains[k]
                    hi=base * cmax
                    np.minimum(arr, hi, out=arr)
        _clip_inplace()
        if self._slew_active:
            tgt = np.array([
                _safe_ratio(self.P_COEFF_TOR, self.default_gains['P']),
                _safe_ratio(self.I_COEFF_TOR, self.default_gains['I']),
                _safe_ratio(self.D_COEFF_TOR, self.default_gains['D'])
            ], dtype=float)
            delta = np.clip(tgt - self._prev_mult, -self._slew, self._slew)
            new_m = self._prev_mult + delta
            self.P_COEFF_TOR = self.default_gains['P'] * new_m[0]
            self.I_COEFF_TOR = self.default_gains['I'] * new_m[1]
            self.D_COEFF_TOR = self.default_gains['D'] * new_m[2]
            self._prev_mult = new_m
        else:
            self._prev_mult = np.array([
                _safe_ratio(self.P_COEFF_TOR, self.default_gains['P']),
                _safe_ratio(self.I_COEFF_TOR, self.default_gains['I']),
                _safe_ratio(self.D_COEFF_TOR, self.default_gains['D'])
            ], dtype=float)
        self.gain_history.append({'P':_avg(self.P_COEFF_TOR),'I':_avg(self.I_COEFF_TOR),'D':_avg(self.D_COEFF_TOR)})
    def get_overlap_metrics(self):
        steps = max(1, int(self._metric_steps))
        mean_overlap = float(self._metric_overlap_sum) / steps
        mean_action_diff = float(self._metric_action_diff_step_sum) / steps
        return {'mean_overlap': mean_overlap, 'mean_action_diff': mean_action_diff}
    def dump_rule_stats(self): return dict(self._segment_hit_counter)
    def get_gain_history(self): return list(self.gain_history)
    def get_last_rule_name(self): return self.last_rule_name
    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_rpy=np.zeros(3), target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        st=self._get_control_state(cur_pos,cur_quat,cur_ang_vel,target_pos,target_rpy)
        self._update_gains(st)
        rpm,pos_e,rpy_e=super().computeControl(control_timestep,cur_pos,cur_quat,cur_vel,cur_ang_vel,target_pos,target_rpy,target_vel,target_rpy_rates)
        return rpm,pos_e,rpy_e
    def _get_control_state(self,cur_pos,cur_quat,cur_ang_vel,target_pos,target_rpy)->Dict[str,float]:
        pos_e=target_pos-cur_pos
        rpy=Rotation.from_quat(cur_quat).as_euler('XYZ',degrees=False)
        rpy_e=target_rpy-rpy
        pos_err_xy = float(np.linalg.norm(pos_e[:2]))
        rpy_err_mag = float(np.linalg.norm(rpy_e))
        ang_vel_mag = float(np.linalg.norm(cur_ang_vel))
        pos_err_z_abs = float(abs(pos_e[2]))
        return {
            'err_p_roll': rpy_e[0], 'err_p_pitch': rpy_e[1], 'err_p_yaw': rpy_e[2],
            'err_d_roll': -cur_ang_vel[0], 'err_d_pitch': -cur_ang_vel[1], 'err_d_yaw': -cur_ang_vel[2],
            'ang_vel_x': cur_ang_vel[0], 'ang_vel_y': cur_ang_vel[1], 'ang_vel_z': cur_ang_vel[2],
            'err_i_roll': self.integral_rpy_e[0], 'err_i_pitch': self.integral_rpy_e[1], 'err_i_yaw': self.integral_rpy_e[2],
            'pos_err_x': pos_e[0], 'pos_err_y': pos_e[1], 'pos_err_z': pos_e[2],
            'err_i_x': self.integral_pos_e[0], 'err_i_y': self.integral_pos_e[1], 'err_i_z': self.integral_pos_e[2],
            'pos_err_xy': pos_err_xy,
            'rpy_err_mag': rpy_err_mag,
            'ang_vel_mag': ang_vel_mag,
            'pos_err_z_abs': pos_err_z_abs,
        }
