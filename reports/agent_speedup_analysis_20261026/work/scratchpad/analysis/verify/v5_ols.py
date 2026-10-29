import json, math
calls = json.load(open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/v5_uc7_calls.json'))
def solve(A, b):
    n = len(A); M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c])); M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1): M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]
def inv(A):
    n = len(A); M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c])); M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [v / piv for v in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]
def ols(X, y):
    n, k = len(X), len(X[0])
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(k)]
    beta = solve(XtX, Xty)
    res = [y[i] - sum(beta[a] * X[i][a] for a in range(k)) for i in range(n)]
    sse = sum(r * r for r in res); my = sum(y) / n; sst = sum((v - my) ** 2 for v in y)
    s2 = sse / (n - k); V = inv(XtX)
    se = [math.sqrt(s2 * V[a][a]) for a in range(k)]
    return beta, se, 1 - sse / sst
y = [c['lat'] for c in calls]
inp = [c['inp'] / 1000 for c in calls]; reas = [c['reas'] for c in calls]; vis = [c['out'] - c['reas'] for c in calls]
for name, X in (('L~input', [[1, a] for a in inp]), ('L~reas', [[1, r] for r in reas]), ('L~out', [[1, c['out']] for c in calls]),
                ('L~input+reas+vis', [[1, a, r, v] for a, r, v in zip(inp, reas, vis)])):
    b, se, r2 = ols(X, y)
    print(name, 'beta', [round(v, 4) for v in b], 'se', [round(v, 4) for v in se], 'R2 %.3f' % r2)
# decomposition at means
b, se, r2 = ols([[1, a, r, v] for a, r, v in zip(inp, reas, vis)], y)
n = len(y)
print('sum latency %.1f' % sum(y))
print('components: intercept %.0f input %.0f reasoning %.0f visible %.0f' % (b[0] * n, b[1] * sum(inp), b[2] * sum(reas), b[3] * sum(vis)))
print('mean input ktok %.1f -> partial input effect per mean call %.2f s; at 154k: %.2f s' % (sum(inp) / n, b[1] * sum(inp) / n, b[1] * 153.9))
