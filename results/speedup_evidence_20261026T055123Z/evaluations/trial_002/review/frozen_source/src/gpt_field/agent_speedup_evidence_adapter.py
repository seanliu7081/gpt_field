"""Private evaluator instrumentation shared by A/B; no policy/controller changes."""
import ast
import inspect
from .agent_speedup_evidence_clock import EvaluationBudgetExceeded


def adapt_service_budget(module, clock):
    source=inspect.getsource(module.execute)
    tree=ast.parse(source)
    counts={'loop':0,'act':0,'step':0,'handler':0}
    class Bind(ast.NodeTransformer):
        def visit_While(self,node):
            self.generic_visit(node)
            if any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='decision' for t in n.targets) for n in node.body):
                node.body[:0]=ast.parse('_service_clock.mark_raw(sim.status.get("libero_success"), step=steps)\n_service_clock.check()').body
                counts['loop']+=1
            return node
        def visit_Assign(self,node):
            self.generic_visit(node)
            if any(isinstance(t,ast.Name) and t.id=='decision' for t in node.targets) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='act':
                node.value=ast.parse('_service_clock.call(policy.act, obs, event="policy_decision")',mode='eval').body
                counts['act']+=1
                return [node,*ast.parse('_service_clock.check()\nif steps == 0: _service_clock.event("first_action_ready")').body]
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='sim' and node.value.func.attr=='step':
                counts['step']+=1
                return [node,*ast.parse('_service_clock.mark_raw(sim.status.get("libero_success"), step=steps+1)').body]
            return node
        def visit_Try(self,node):
            self.generic_visit(node)
            if any(isinstance(h.type,ast.Name) and h.type.id=='KeyboardInterrupt' for h in node.handlers):
                handler=ast.parse('''try:
    pass
except EvaluationBudgetExceeded as exc:
    status, caught, interrupted_stage = "evaluation_budget_exhausted", exc, stage
    error = {"type": "EvaluationBudgetExceeded", "stage": stage, "source": "effective_service_clock", "message": str(exc)}
''').body[0].handlers[0]
                node.handlers.insert(0,handler);counts['handler']+=1
            return node
    Bind().visit(tree)
    if counts!={'loop':1,'act':1,'step':1,'handler':1}: raise RuntimeError(f'evaluator AST changed: {counts}')
    ast.fix_missing_locations(tree);module._service_clock=clock;module.EvaluationBudgetExceeded=EvaluationBudgetExceeded
    # Set a distinct filename: subsequent IO AST adaptation must inspect this
    # instrumented source rather than re-read the historical function from disk.
    import linecache
    text=ast.unparse(tree)+'\n';filename=module.__file__+'::service_clock'
    linecache.cache[filename]=(len(text),None,text.splitlines(True),filename)
    exec(compile(text,filename,'exec'),module.__dict__)
    return counts
