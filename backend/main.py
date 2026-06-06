from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sympy import (symbols, sympify, diff, latex, simplify,
                   Matrix, lambdify)
import numpy as np
import math

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
x, y = symbols('x y')
from sympy import hessian, Function, cos, sin, exp as sym_exp

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def _digit_char(d):
    return "0123456789ABCDEF"[d]

def _dec_to_base_str(val, base, prec=10):
    neg = val < 0
    val = abs(val)
    ei = int(val)
    ef = val - ei
    chars = []
    n = ei
    if n == 0:
        chars = ["0"]
    while n > 0:
        chars.append(_digit_char(n % base))
        n //= base
    entera = "".join(reversed(chars))
    frac = ""
    tmp = ef
    for _ in range(prec):
        if tmp < 1e-12:
            break
        tmp *= base
        d_ = int(tmp)
        frac += _digit_char(d_)
        tmp -= d_
    return ("-" if neg else "") + entera + ("." + frac if frac else "")

# ──────────────────────────────────────────────────────────────
# TAYLOR  (original conservado tal cual)
# ──────────────────────────────────────────────────────────────
@app.get("/taylor")
def taylor(expr: str, a: float = 0, n: int = 5):
    try:
        f = sympify(expr)
        pasos = []
        pasos.append("Función ingresada:")
        pasos.append(f"f(x) = {str(f)}")
        f_a = f.subs(x, a).evalf()
        pasos.append("\nPaso 1: Evaluar la función en a")
        pasos.append(f"f({a}) = {f_a}")
        pasos.append("\nPaso 2: Calcular derivadas en a")
        for i in range(n + 1):
            d = diff(f, x, i)
            val = round(float(d.subs(x, a)), 6)
            nombre = "f(x)" if i == 0 else ("f'(x)" if i == 1 else f"f^{i}(x)")
            pasos.append(f"{nombre} en x={a} → {val}")
        taylor_poly = sum(
            (diff(f, x, k).subs(x, a) / math.factorial(k)) * (x - a)**k
            for k in range(n + 1)
        )
        taylor_poly = simplify(taylor_poly.expand())
        pasos.append("\nPaso 3: Construir el polinomio de Taylor")
        pasos.append(str(taylor_poly).replace("**", "^"))
        taylor_latex_str = latex(taylor_poly)
        xs = np.linspace(a - 2, a + 2, 100)
        f_lam = lambdify(x, f, "numpy")
        t_lam = lambdify(x, taylor_poly, "numpy")
        try:
            real_vals   = [float(v) for v in f_lam(xs)]
            approx_vals = [float(v) for v in t_lam(xs)]
        except Exception:
            real_vals   = [float(f.subs(x, v).evalf()) for v in xs]
            approx_vals = [float(taylor_poly.subs(x, v).evalf()) for v in xs]
        error_vals = [abs(r - t) for r, t in zip(real_vals, approx_vals)]
        return {
            "success": True, "expr": str(f), "pasos": pasos,
            "taylor": str(taylor_poly).replace("**", "^"),
            "taylor_latex": taylor_latex_str,
            "x": xs.tolist(), "real": real_vals,
            "approx": approx_vals, "error": error_vals,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ──────────────────────────────────────────────────────────────
# CAMBIO DE BASE
# ──────────────────────────────────────────────────────────────
@app.get("/base")
def cambio_base(numero: str, base_origen: int = 10, base_destino: int = 2):
    try:
        pasos = []
        num_upper = numero.strip().upper()
        # Convertir a decimal
        pasos.append("Paso 1: Convertir el número a decimal (base 10)")
        if base_origen == 10:
            dec_val = float(num_upper)
            pasos.append(f"El número ya está en base 10: {dec_val}")
        else:
            partes = num_upper.split(".")
            parte_entera = int(partes[0], base_origen)
            parte_frac = 0.0
            pasos.append(f"Parte entera '{partes[0]}' en base {base_origen}:")
            digitos = list(partes[0])
            detalle = " + ".join(
                f"{d}×{base_origen}^{len(digitos)-1-i}" for i, d in enumerate(digitos)
            )
            pasos.append(f"  {detalle} = {parte_entera}")
            if len(partes) > 1:
                pasos.append(f"Parte fraccionaria '{partes[1]}' en base {base_origen}:")
                for i, d in enumerate(partes[1]):
                    val = int(d, base_origen) / (base_origen ** (i + 1))
                    parte_frac += val
                    pasos.append(f"  {d}×{base_origen}^(-{i+1}) = {round(val,8)}")
            dec_val = parte_entera + parte_frac
            pasos.append(f"Valor decimal total: {dec_val}")
        # Convertir decimal → base destino
        pasos.append(f"\nPaso 2: Convertir {dec_val} de decimal a base {base_destino}")
        ei = int(abs(dec_val))
        ef = abs(dec_val) - ei
        residuos = []
        n_tmp = ei
        pasos.append("Parte entera (divisiones sucesivas):")
        if n_tmp == 0:
            residuos = [0]
            pasos.append(f"  0 ÷ {base_destino} = 0 resto 0 → {_digit_char(0)}")
        while n_tmp > 0:
            r = n_tmp % base_destino
            pasos.append(f"  {n_tmp} ÷ {base_destino} = {n_tmp//base_destino} resto {r} → {_digit_char(r)}")
            residuos.append(r)
            n_tmp //= base_destino
        residuos.reverse()
        entera_str = "".join(_digit_char(r) for r in residuos)
        pasos.append(f"Leer residuos de abajo hacia arriba: {entera_str}")
        frac_str = ""
        if ef > 1e-12:
            pasos.append("Parte fraccionaria (multiplicaciones sucesivas):")
            tmp = ef
            for _ in range(10):
                if tmp < 1e-12:
                    break
                tmp *= base_destino
                d_ = int(tmp)
                pasos.append(f"  {round(tmp/base_destino,8)} × {base_destino} = {round(tmp,8)} → {_digit_char(d_)}")
                frac_str += _digit_char(d_)
                tmp -= d_
            pasos.append(f"Parte fraccionaria en base {base_destino}: .{frac_str}")
        resultado = ("-" if dec_val < 0 else "") + entera_str + ("." + frac_str if frac_str else "")
        pasos.append(f"\nResultado: {numero} (base {base_origen}) = {resultado} (base {base_destino})")
        resumen = {str(b): _dec_to_base_str(dec_val, b) for b in [2, 8, 10, 16]}
        return {
            "success": True,
            "numero_original": numero,
            "base_origen": base_origen,
            "base_destino": base_destino,
            "decimal": dec_val,
            "resultado": resultado,
            "pasos": pasos,
            "resumen": resumen,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ──────────────────────────────────────────────────────────────
# BISECCIÓN
# ──────────────────────────────────────────────────────────────
@app.get("/biseccion")
def biseccion(expr: str, a: float, b: float, tol: float = 1e-4, max_iter: int = 100):
    try:
        f_sym = sympify(expr)
        pasos = []
        pasos.append(f"Función: f(x) = {expr}")
        pasos.append(f"Intervalo inicial: [{a}, {b}]  |  Tolerancia: {tol}")
        fa = float(f_sym.subs(x, a).evalf())
        fb = float(f_sym.subs(x, b).evalf())
        pasos.append(f"\nVerificación cambio de signo:")
        pasos.append(f"f({a}) = {round(fa,6)}")
        pasos.append(f"f({b}) = {round(fb,6)}")
        if fa * fb >= 0:
            return {"success": False, "error": "f(a) y f(b) deben tener signos opuestos"}
        pasos.append(f"f(a)·f(b) = {round(fa*fb,6)} < 0  ✓")
        ai, bi = a, b
        tabla = []
        c = None
        for i in range(max_iter):
            c = (ai + bi) / 2
            fc = float(f_sym.subs(x, c).evalf())
            err = (bi - ai) / 2
            fai = float(f_sym.subs(x, ai).evalf())
            tabla.append({"n": i+1, "a": round(ai,8), "b": round(bi,8),
                          "c": round(c,8), "fc": round(fc,8), "error": round(err,8)})
            pasos.append(f"\nPaso {i+1}: c=({round(ai,5)}+{round(bi,5)})/2 = {round(c,8)},  f(c)={round(fc,8)},  error={round(err,8)}")
            if abs(fc) < 1e-14 or err < tol:
                pasos.append(f"Convergió en iteración {i+1}  ✓")
                break
            if fai * fc < 0:
                bi = c
                pasos.append(f"  → raíz en [{round(ai,5)}, {round(c,5)}] — actualizar b = c")
            else:
                ai = c
                pasos.append(f"  → raíz en [{round(c,5)}, {round(bi,5)}] — actualizar a = c")
        xs_p = np.linspace(a - abs(b-a)*0.3, b + abs(b-a)*0.3, 200).tolist()
        ys_p = [float(f_sym.subs(x, v).evalf()) for v in xs_p]
        return {
            "success": True,
            "raiz": round(c, 10),
            "f_raiz": round(float(f_sym.subs(x, c).evalf()), 10),
            "iteraciones_total": len(tabla),
            "pasos": pasos, "tabla": tabla,
            "x_plot": xs_p, "y_plot": ys_p, "raiz_x": c,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ──────────────────────────────────────────────────────────────
# NEWTON-RAPHSON (1 variable)
# ──────────────────────────────────────────────────────────────
@app.get("/newton")
def newton(expr: str, x0: float, tol: float = 1e-4, max_iter: int = 50):
    try:
        f_sym  = sympify(expr)
        df_sym = diff(f_sym, x)
        pasos = []
        pasos.append(f"Función:  f(x)  = {expr}")
        pasos.append(f"Derivada: f'(x) = {str(df_sym)}")
        pasos.append(f"x₀ = {x0}  |  Tolerancia: {tol}")
        pasos.append(f"\nFórmula: x_{{n+1}} = x_n - f(x_n) / f'(x_n)")
        tabla = []
        xn = x0
        for i in range(max_iter):
            fxn  = float(f_sym.subs(x, xn).evalf())
            dfxn = float(df_sym.subs(x, xn).evalf())
            if abs(dfxn) < 1e-14:
                return {"success": False, "error": f"f'(x) ≈ 0 en x={xn}, el método no converge"}
            x1  = xn - fxn / dfxn
            err = abs(x1 - xn)
            tabla.append({"n": i+1, "xn": round(xn,8), "fxn": round(fxn,8),
                           "dfxn": round(dfxn,8), "x1": round(x1,8), "error": round(err,8)})
            pasos.append(f"\nPaso {i+1}: x = {round(xn,8)} - ({round(fxn,8)})/({round(dfxn,8)}) = {round(x1,8)},  error={round(err,8)}")
            xn = x1
            if err < tol:
                pasos.append(f"Convergió en iteración {i+1}  ✓")
                break
        xs_p = np.linspace(x0 - 3, x0 + 3, 300).tolist()
        ys_p = []
        for v in xs_p:
            try:
                ys_p.append(float(f_sym.subs(x, v).evalf()))
            except Exception:
                ys_p.append(None)
        return {
            "success": True,
            "raiz": round(xn, 10),
            "f_raiz": round(float(f_sym.subs(x, xn).evalf()), 10),
            "derivada_latex": latex(df_sym),
            "iteraciones_total": len(tabla),
            "pasos": pasos, "tabla": tabla,
            "x_plot": xs_p, "y_plot": ys_p, "raiz_x": xn,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ──────────────────────────────────────────────────────────────
# SISTEMA NO LINEAL 2×2 — Newton
# ──────────────────────────────────────────────────────────────
@app.get("/sistema")
def sistema_nl(f1: str, f2: str, x0: float = 1.0, y0: float = 1.0,
               tol: float = 1e-5, max_iter: int = 30):
    try:
        f1s = sympify(f1)
        f2s = sympify(f2)
        J   = Matrix([[diff(f1s, x), diff(f1s, y)],
                      [diff(f2s, x), diff(f2s, y)]])
        pasos = []
        pasos.append(f"f₁(x,y) = {f1}")
        pasos.append(f"f₂(x,y) = {f2}")
        pasos.append(f"\nJacobiana (calculada simbólicamente):")
        pasos.append(f"  ∂f₁/∂x = {str(diff(f1s,x))}")
        pasos.append(f"  ∂f₁/∂y = {str(diff(f1s,y))}")
        pasos.append(f"  ∂f₂/∂x = {str(diff(f2s,x))}")
        pasos.append(f"  ∂f₂/∂y = {str(diff(f2s,y))}")
        pasos.append(f"\nFórmula: [x,y]_{{k+1}} = [x,y]_k - J⁻¹·F")
        pasos.append(f"Punto inicial: ({x0}, {y0})")
        tabla = []
        xn, yn = x0, y0
        for i in range(max_iter):
            s = {x: xn, y: yn}
            F = Matrix([float(f1s.subs(s).evalf()), float(f2s.subs(s).evalf())])
            Jv = J.subs(s).evalf()
            det = float(Jv.det())
            if abs(det) < 1e-14:
                return {"success": False, "error": f"Jacobiana singular (det≈0) en iteración {i+1}"}
            delta = Jv.inv() * (-F)
            dx_, dy_ = float(delta[0]), float(delta[1])
            norm = math.sqrt(dx_**2 + dy_**2)
            tabla.append({"n": i+1, "x": round(xn,8), "y": round(yn,8),
                           "f1": round(float(F[0]),8), "f2": round(float(F[1]),8), "norm": round(norm,8)})
            pasos.append(f"\nPaso {i+1}: ({round(xn,6)},{round(yn,6)})  F=({round(float(F[0]),6)},{round(float(F[1]),6)})  ‖Δ‖={round(norm,8)}")
            xn += dx_
            yn += dy_
            if norm < tol:
                pasos.append(f"Convergió en iteración {i+1}  ✓")
                break
        return {
            "success": True,
            "x": round(xn, 10), "y": round(yn, 10),
            "f1_val": round(float(f1s.subs({x:xn, y:yn}).evalf()), 10),
            "f2_val": round(float(f2s.subs({x:xn, y:yn}).evalf()), 10),
            "iteraciones_total": len(tabla),
            "jacobiana_latex": latex(J),
            "pasos": pasos, "tabla": tabla,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ──────────────────────────────────────────────────────────────
# LAGRANGE
# ──────────────────────────────────────────────────────────────
@app.get("/lagrange")
def lagrange(xs_str: str, ys_str: str, eval_x: float = None):
    try:
        xs_list = [float(v) for v in xs_str.split(",")]
        ys_list = [float(v) for v in ys_str.split(",")]
        if len(xs_list) != len(ys_list):
            return {"success": False, "error": "xs e ys deben tener la misma cantidad de puntos"}
        n = len(xs_list)
        pasos = []
        pasos.append(f"Puntos: {list(zip([round(v,4) for v in xs_list], [round(v,4) for v in ys_list]))}")
        pasos.append(f"Grado del polinomio: {n-1}")
        pasos.append("\nPaso 1: Calcular las bases de Lagrange Lᵢ(x)")
        P = 0
        for i in range(n):
            Li = 1
            for j in range(n):
                if i != j:
                    Li *= (x - xs_list[j]) / (xs_list[i] - xs_list[j])
            Li_s = simplify(Li)
            P += ys_list[i] * Li_s
            expr_str = " · ".join(
                f"(x-{xs_list[j]})/({xs_list[i]}-{xs_list[j]})"
                for j in range(n) if j != i
            )
            pasos.append(f"\nL_{i}(x) = {expr_str}")
            pasos.append(f"  simplificado: {str(Li_s).replace('**','^')}")
            if eval_x is not None:
                li_val = float(Li_s.subs(x, eval_x).evalf())
                pasos.append(f"  L_{i}({eval_x}) = {round(li_val,6)}  →  {ys_list[i]} × {round(li_val,6)} = {round(ys_list[i]*li_val,6)}")
        P_simplified = simplify(P.expand())
        pasos.append(f"\nPaso 2: P(x) = Σ yᵢ·Lᵢ(x)")
        pasos.append(f"P(x) = {str(P_simplified).replace('**','^')}")
        resultado_eval = None
        if eval_x is not None:
            resultado_eval = round(float(P_simplified.subs(x, eval_x).evalf()), 8)
            pasos.append(f"\nPaso 3: Evaluar en x = {eval_x}")
            pasos.append(f"P({eval_x}) = {resultado_eval}")
        xs_p = np.linspace(min(xs_list)-0.5, max(xs_list)+0.5, 200).tolist()
        P_lam = lambdify(x, P_simplified, "numpy")
        try:
            ys_p = [float(P_lam(v)) for v in xs_p]
        except Exception:
            ys_p = [float(P_simplified.subs(x, v).evalf()) for v in xs_p]
        return {
            "success": True,
            "polinomio": str(P_simplified).replace("**","^"),
            "polinomio_latex": latex(P_simplified),
            "eval_x": eval_x,
            "resultado_eval": resultado_eval,
            "pasos": pasos,
            "x_plot": xs_p, "y_plot": ys_p,
            "puntos_x": xs_list, "puntos_y": ys_list,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ──────────────────────────────────────────────────────────────
# DIFERENCIAS DIVIDIDAS DE NEWTON
# ──────────────────────────────────────────────────────────────
@app.get("/diferencias")
def diferencias_divididas(xs_str: str, ys_str: str, eval_x: float = None):
    try:
        xs_list = [float(v) for v in xs_str.split(",")]
        ys_list = [float(v) for v in ys_str.split(",")]
        if len(xs_list) != len(ys_list):
            return {"success": False, "error": "xs e ys deben tener la misma cantidad de puntos"}
        n = len(xs_list)
        # Tabla de diferencias divididas
        F = [[0.0]*n for _ in range(n)]
        for i in range(n):
            F[i][0] = ys_list[i]
        for j in range(1, n):
            for i in range(n - j):
                F[i][j] = (F[i+1][j-1] - F[i][j-1]) / (xs_list[i+j] - xs_list[i])
        coefs = [F[0][k] for k in range(n)]
        pasos = []
        pasos.append(f"Puntos: {list(zip([round(v,4) for v in xs_list],[round(v,4) for v in ys_list]))}")
        pasos.append(f"\nPaso 1: Construir la tabla de diferencias divididas")
        header = "i  |  x     |  f[·]  " + "".join(f"|  Ord.{k}  " for k in range(1,n))
        pasos.append(header)
        pasos.append("-"*len(header))
        for i in range(n):
            fila = f"{i}  |  {round(xs_list[i],4):<6}|  {round(F[i][0],6):<8}"
            for j in range(1, n-i):
                fila += f"|  {round(F[i][j],6):<8}"
            pasos.append(fila)
        pasos.append(f"\nPaso 2: Coeficientes (diagonal superior):")
        pasos.append("  " + ", ".join(f"f[x₀..x{k}]={round(coefs[k],6)}" for k in range(n)))
        pasos.append(f"\nPaso 3: Construir el polinomio")
        pasos.append("P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ...")
        # Construir simbólicamente
        P = 0
        prod = 1
        for k in range(n):
            P += coefs[k] * prod
            prod *= (x - xs_list[k])
        P_simplified = simplify(P.expand())
        pasos.append(f"P(x) = {str(P_simplified).replace('**','^')}")
        resultado_eval = None
        if eval_x is not None:
            resultado_eval = round(float(P_simplified.subs(x, eval_x).evalf()), 8)
            pasos.append(f"\nPaso 4: Evaluar en x = {eval_x}")
            prod_val = 1.0
            acum = coefs[0]
            pasos.append(f"  P({eval_x}) = {round(coefs[0],6)}")
            for k in range(1, n):
                prod_val *= (eval_x - xs_list[k-1])
                term = coefs[k] * prod_val
                acum += term
                pasos.append(f"  + {round(coefs[k],6)} × {round(prod_val,6)} = {round(term,6)}  (acumulado: {round(acum,6)})")
            pasos.append(f"P({eval_x}) = {resultado_eval}")
        tabla_dd = [{"i": i, "x": xs_list[i], "orden0": round(F[i][0],6),
                     **{f"orden{j}": round(F[i][j],6) for j in range(1, n-i)}}
                    for i in range(n)]
        xs_p = np.linspace(min(xs_list)-0.5, max(xs_list)+0.5, 200).tolist()
        P_lam = lambdify(x, P_simplified, "numpy")
        try:
            ys_p = [float(P_lam(v)) for v in xs_p]
        except Exception:
            ys_p = [float(P_simplified.subs(x, v).evalf()) for v in xs_p]
        return {
            "success": True,
            "coeficientes": [round(c, 8) for c in coefs],
            "polinomio": str(P_simplified).replace("**","^"),
            "polinomio_latex": latex(P_simplified),
            "eval_x": eval_x,
            "resultado_eval": resultado_eval,
            "pasos": pasos,
            "tabla_dd": tabla_dd,
            "x_plot": xs_p, "y_plot": ys_p,
            "puntos_x": xs_list, "puntos_y": ys_list,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ──────────────────────────────────────────────────────────────
# TAYLOR 2 VARIABLES
# ──────────────────────────────────────────────────────────────
@app.get("/taylor2")
def taylor2(expr: str, a: float = 0, b: float = 0, orden: int = 2,
            eval_x: float = None, eval_y: float = None):
    try:
        import numpy as np
        f = sympify(expr)
        pasos = []
        pasos.append(f"Función: f(x,y) = {str(f)}")
        pasos.append(f"Punto de expansión: (a={a}, b={b}),  Orden: {orden}")
        s0 = {x: a, y: b}
        f0 = float(f.subs(s0).evalf())
        fx  = diff(f, x);  fx_v  = float(fx.subs(s0).evalf())
        fy  = diff(f, y);  fy_v  = float(fy.subs(s0).evalf())
        pasos.append(f"\nPaso 1: Derivadas de primer orden en ({a},{b})")
        pasos.append(f"  f(a,b)  = {round(f0,8)}")
        pasos.append(f"  fx(a,b) = {round(fx_v,8)}   [∂f/∂x = {str(fx)}]")
        pasos.append(f"  fy(a,b) = {round(fy_v,8)}   [∂f/∂y = {str(fy)}]")
        dx_sym = x - a;  dy_sym = y - b
        P = f0 + fx_v * dx_sym + fy_v * dy_sym
        fxx_v = fxy_v = fyy_v = None
        if orden >= 2:
            fxx = diff(f, x, 2);   fxx_v = float(fxx.subs(s0).evalf())
            fxy = diff(diff(f,x),y); fxy_v = float(fxy.subs(s0).evalf())
            fyy = diff(f, y, 2);   fyy_v = float(fyy.subs(s0).evalf())
            pasos.append(f"\nPaso 2: Derivadas de segundo orden en ({a},{b})")
            pasos.append(f"  fxx(a,b) = {round(fxx_v,8)}   [∂²f/∂x²  = {str(fxx)}]")
            pasos.append(f"  fxy(a,b) = {round(fxy_v,8)}   [∂²f/∂x∂y = {str(diff(diff(f,x),y))}]")
            pasos.append(f"  fyy(a,b) = {round(fyy_v,8)}   [∂²f/∂y²  = {str(fyy)}]")
            P += (fxx_v/2)*dx_sym**2 + fxy_v*dx_sym*dy_sym + (fyy_v/2)*dy_sym**2
        P_sym = simplify(P)
        P_lat = latex(P_sym)
        pasos.append(f"\nPaso 3: Polinomio P(x,y) de orden {orden}:")
        pasos.append(f"  P(x,y) = {str(P_sym).replace('**','^')}")
        resultado_eval = real_val = error_abs = None
        if eval_x is not None and eval_y is not None:
            resultado_eval = float(P_sym.subs({x: eval_x, y: eval_y}).evalf())
            real_val       = float(f.subs({x: eval_x, y: eval_y}).evalf())
            error_abs      = abs(real_val - resultado_eval)
            pasos.append(f"\nPaso 4: Evaluación en ({eval_x}, {eval_y})")
            pasos.append(f"  P({eval_x},{eval_y}) = {round(resultado_eval,8)}")
            pasos.append(f"  f({eval_x},{eval_y}) = {round(real_val,8)}  (valor real)")
            pasos.append(f"  Error absoluto      = {error_abs:.6e}")
        xs_g = np.linspace(a-1.5, a+1.5, 20).tolist()
        ys_g = np.linspace(b-1.5, b+1.5, 20).tolist()
        f_lam = lambdify([x, y], f, "numpy")
        P_lam = lambdify([x, y], P_sym, "numpy")
        try:
            Z_real = [[float(f_lam(xi,yi)) for xi in xs_g] for yi in ys_g]
            Z_poly = [[float(P_lam(xi,yi)) for xi in xs_g] for yi in ys_g]
        except Exception:
            Z_real = [[float(f.subs({x:xi,y:yi}).evalf())   for xi in xs_g] for yi in ys_g]
            Z_poly = [[float(P_sym.subs({x:xi,y:yi}).evalf()) for xi in xs_g] for yi in ys_g]
        return {
            "success": True, "expr": str(f),
            "polinomio": str(P_sym).replace("**","^"), "polinomio_latex": P_lat,
            "f0":f0,"fx":fx_v,"fy":fy_v,"fxx":fxx_v,"fxy":fxy_v,"fyy":fyy_v,
            "eval_x":eval_x,"eval_y":eval_y,
            "resultado_eval": round(resultado_eval,8) if resultado_eval is not None else None,
            "real_val": round(real_val,8) if real_val is not None else None,
            "error_abs": error_abs, "pasos": pasos,
            "xs_grid":xs_g,"ys_grid":ys_g,"z_real":Z_real,"z_poly":Z_poly,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ──────────────────────────────────────────────────────────────
# AJUSTE DE CURVAS
# ──────────────────────────────────────────────────────────────
def _gauss_elim(A, b_vec):
    import numpy as np
    n = len(b_vec)
    M = np.array([[A[i][j] for j in range(n)] + [b_vec[i]] for i in range(n)], dtype=float)
    for col in range(n):
        piv = max(range(col,n), key=lambda r: abs(M[r][col]))
        M[[col,piv]] = M[[piv,col]]
        if abs(M[col][col]) < 1e-14:
            raise ValueError("Sistema singular — puntos colineales o insuficientes para el grado")
        for row in range(col+1,n):
            f_ = M[row][col]/M[col][col]; M[row] -= f_*M[col]
    sol = np.zeros(n)
    for i in range(n-1,-1,-1):
        sol[i] = M[i][n]
        for j in range(i+1,n): sol[i] -= M[i][j]*sol[j]
        sol[i] /= M[i][i]
    return sol.tolist()

def _r2(ys, yp):
    import numpy as np
    ys,yp=np.array(ys),np.array(yp)
    ss=np.sum((ys-ys.mean())**2)
    return float(1-np.sum((ys-yp)**2)/ss) if ss>0 else 1.0

@app.get("/ajuste/lineal")
def ajuste_lineal(xs_str: str, ys_str: str, eval_x: float = None):
    try:
        import numpy as np
        xs=[float(v) for v in xs_str.split(",")]; ys=[float(v) for v in ys_str.split(",")]
        if len(xs)!=len(ys): raise ValueError("xs e ys deben tener la misma cantidad de puntos")
        n=len(xs)
        if n<2: raise ValueError("Se necesitan al menos 2 puntos")
        sx=sum(xs);sy=sum(ys);sxy=sum(xs[i]*ys[i] for i in range(n));sx2=sum(xi**2 for xi in xs)
        det=n*sx2-sx**2
        if abs(det)<1e-14: raise ValueError("Todos los x son iguales — sistema singular")
        a1=(n*sxy-sx*sy)/det; a0=(sy-a1*sx)/n
        yp=[a0+a1*xi for xi in xs]; R2=_r2(ys,yp)
        pasos=[
            f"Puntos: {list(zip([round(v,4) for v in xs],[round(v,4) for v in ys]))}",
            f"Modelo: y = a₀ + a₁·x",
            f"\nSumas:  Σx={round(sx,6)}  Σy={round(sy,6)}  Σxy={round(sxy,6)}  Σx²={round(sx2,6)}",
            f"\nPaso 1: Coeficientes",
            f"  a₁ = (n·Σxy − Σx·Σy)/(n·Σx² − (Σx)²) = {round(a1,8)}",
            f"  a₀ = (Σy − a₁·Σx)/n = {round(a0,8)}",
            f"\nModelo: y = {round(a0,6)} + {round(a1,6)}·x",
            f"R² = {round(R2,6)}  ({round(R2*100,2)}% varianza explicada)",
            f"\nResiduos:",
        ]
        for i in range(n): pasos.append(f"  x={xs[i]}, y={ys[i]}, ŷ={round(yp[i],6)}, e={round(ys[i]-yp[i],6)}")
        resultado_eval=None
        if eval_x is not None:
            resultado_eval=round(a0+a1*eval_x,8); pasos.append(f"\ny({eval_x}) = {resultado_eval}")
        xs_p=np.linspace(min(xs)-.5,max(xs)+.5,100).tolist()
        return {"success":True,"a0":round(a0,8),"a1":round(a1,8),"R2":round(R2,6),
                "resultado_eval":resultado_eval,"pasos":pasos,
                "x_plot":xs_p,"y_plot":[a0+a1*xi for xi in xs_p],"puntos_x":xs,"puntos_y":ys}
    except Exception as e: return {"success":False,"error":str(e)}

@app.get("/ajuste/polinomial")
def ajuste_polinomial(xs_str: str, ys_str: str, grado: int = 2, eval_x: float = None):
    try:
        import numpy as np
        xs=[float(v) for v in xs_str.split(",")]; ys=[float(v) for v in ys_str.split(",")]
        if len(xs)!=len(ys): raise ValueError("xs e ys deben tener la misma cantidad de puntos")
        if grado not in (2,3): raise ValueError("Grado debe ser 2 o 3")
        if len(xs)<=grado: raise ValueError(f"Se necesitan al menos {grado+1} puntos para grado {grado}")
        n=len(xs); m=grado+1
        A=[[xi**k for k in range(m)] for xi in xs]
        At=[[A[i][j] for i in range(n)] for j in range(m)]
        AtA=[[sum(At[i][k]*At[j][k] for k in range(n)) for j in range(m)] for i in range(m)]
        Aty=[sum(At[i][k]*ys[k] for k in range(n)) for i in range(m)]
        coefs=_gauss_elim(AtA,Aty)
        def poly(xv): return sum(coefs[k]*xv**k for k in range(m))
        yp=[poly(xi) for xi in xs]; R2=_r2(ys,yp)
        terms=" + ".join(
            f"{round(coefs[k],6)}" if k==0 else
            (f"{round(coefs[k],6)}·x" if k==1 else f"{round(coefs[k],6)}·x^{k}")
            for k in range(m))
        pasos=[
            f"Puntos: {list(zip([round(v,4) for v in xs],[round(v,4) for v in ys]))}",
            f"Grado: {grado}",
            f"\nSistema normal A^T·A·c = A^T·y  (eliminación gaussiana con pivoteo)",
            f"\nCoeficientes:",
        ]
        for k,c in enumerate(coefs): pasos.append(f"  a{k} = {round(c,8)}")
        pasos+=[f"\nModelo: y = {terms}",
                f"R² = {round(R2,6)}  ({round(R2*100,2)}% varianza explicada)",
                f"\nResiduos:"]
        for i in range(n): pasos.append(f"  x={xs[i]}, y={ys[i]}, ŷ={round(yp[i],6)}, e={round(ys[i]-yp[i],6)}")
        resultado_eval=None
        if eval_x is not None:
            resultado_eval=round(poly(eval_x),8); pasos.append(f"\ny({eval_x}) = {resultado_eval}")
        xs_p=np.linspace(min(xs)-.5,max(xs)+.5,150).tolist()
        return {"success":True,"coeficientes":[round(c,8) for c in coefs],"grado":grado,"R2":round(R2,6),
                "resultado_eval":resultado_eval,"pasos":pasos,
                "x_plot":xs_p,"y_plot":[poly(xi) for xi in xs_p],"puntos_x":xs,"puntos_y":ys}
    except Exception as e: return {"success":False,"error":str(e)}

@app.get("/ajuste/exponencial")
def ajuste_exponencial(xs_str: str, ys_str: str, eval_x: float = None):
    try:
        import numpy as np, math as _math
        xs=[float(v) for v in xs_str.split(",")]; ys=[float(v) for v in ys_str.split(",")]
        if len(xs)!=len(ys): raise ValueError("xs e ys deben tener la misma cantidad de puntos")
        if any(yi<=0 for yi in ys): raise ValueError("Todos los valores de y deben ser > 0")
        n=len(xs)
        if n<2: raise ValueError("Se necesitan al menos 2 puntos")
        lny=[_math.log(yi) for yi in ys]
        sx=sum(xs);slny=sum(lny);sxlny=sum(xs[i]*lny[i] for i in range(n));sx2=sum(xi**2 for xi in xs)
        det=n*sx2-sx**2
        if abs(det)<1e-14: raise ValueError("Sistema singular")
        b=(n*sxlny-sx*slny)/det; lna=(slny-b*sx)/n; a=_math.exp(lna)
        yp=[a*_math.exp(b*xi) for xi in xs]; R2=_r2(ys,yp)
        pasos=[
            f"Puntos: {list(zip([round(v,4) for v in xs],[round(v,4) for v in ys]))}",
            f"Modelo: y = a·e^(b·x)",
            f"\nLinealización: ln(y) = ln(a) + b·x",
            f"\nSumas:  Σx={round(sx,6)}  Σln(y)={round(slny,6)}  Σx·ln(y)={round(sxlny,6)}  Σx²={round(sx2,6)}",
            f"\nCoeficientes:",
            f"  b    = {round(b,8)}",
            f"  ln(a)= {round(lna,8)}",
            f"  a    = {round(a,8)}",
            f"\nModelo: y = {round(a,6)} · e^({round(b,6)}·x)",
            f"R² = {round(R2,6)}  ({round(R2*100,2)}% varianza explicada)",
            f"\nResiduos:",
        ]
        for i in range(n): pasos.append(f"  x={xs[i]}, y={ys[i]}, ŷ={round(yp[i],6)}, e={round(ys[i]-yp[i],6)}")
        resultado_eval=None
        if eval_x is not None:
            resultado_eval=round(a*_math.exp(b*eval_x),8); pasos.append(f"\ny({eval_x}) = {resultado_eval}")
        xs_p=np.linspace(min(xs)-.5,max(xs)+.5,150).tolist()
        return {"success":True,"a":round(a,8),"b":round(b,8),"R2":round(R2,6),
                "resultado_eval":resultado_eval,"pasos":pasos,
                "x_plot":xs_p,"y_plot":[a*_math.exp(b*xi) for xi in xs_p],"puntos_x":xs,"puntos_y":ys}
    except Exception as e: return {"success":False,"error":str(e)}

# ──────────────────────────────────────────────────────────────
# MÉTODO DE EULER (EDO)
# ──────────────────────────────────────────────────────────────
@app.get("/euler")
def metodo_euler(expr: str, x0: float, y0: float, h: float, n_steps: int = 10):
    try:
        if h <= 0:
            return {"success": False, "error": "El paso h debe ser > 0"}
        if n_steps < 1:
            return {"success": False, "error": "Se necesita al menos 1 paso"}

        f_sym = sympify(expr)
        pasos = [
            f"EDO: dy/dx = f(x,y) = {expr}",
            f"Condición inicial: y({x0}) = {y0}",
            f"Paso h = {h}  |  Pasos: {n_steps}",
            "\nFórmula: y_{i+1} = y_i + h · f(x_i, y_i)",
        ]

        xi, yi = float(x0), float(y0)
        xs_p, ys_p = [xi], [yi]
        tabla = []

        for i in range(n_steps):
            fval = float(f_sym.subs({x: xi, y: yi}).evalf())
            yi1 = yi + h * fval
            xi1 = xi + h

            tabla.append({
                "n": i + 1,
                "xi": round(xi, 8),
                "yi": round(yi, 8),
                "f": round(fval, 8),
                "yi1": round(yi1, 8),
                "xi1": round(xi1, 8),
            })
            pasos.append(
                f"\nPaso {i+1}: f({round(xi,5)}, {round(yi,5)}) = {round(fval,8)}"
            )
            pasos.append(
                f"  y_{i+1} = {round(yi,5)} + {h}·{round(fval,8)} = {round(yi1,8)}"
            )
            pasos.append(f"  x_{i+1} = {round(xi1,8)}")

            xi, yi = xi1, yi1
            xs_p.append(xi)
            ys_p.append(yi)

        return {
            "success": True,
            "x_final": round(xi, 10),
            "y_final": round(yi, 10),
            "iteraciones_total": n_steps,
            "pasos": pasos,
            "tabla": tabla,
            "x_plot": xs_p,
            "y_plot": ys_p,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ──────────────────────────────────────────────────────────────
# DIFERENCIAS FINITAS — BVP
@app.get("/diferencias_finitas")
def diferencias_finitas(
    p_expr: str = "0",    # coeficiente de y'
    q_expr: str = "0",    # coeficiente de y
    r_expr: str = "0",    # lado derecho
    a: float = 0,         # extremo izquierdo
    b: float = 1,         # extremo derecho
    ya: float = 0,        # condición y(a) = ya
    yb: float = 1,        # condición y(b) = yb
    n: int = 10           # número de subintervalos
):
    try:
        if n < 2:
            return {"success": False, "error": "Se necesitan al menos 2 subintervalos"}
        if a >= b:
            return {"success": False, "error": "a debe ser menor que b"}

        p_sym = sympify(p_expr)
        q_sym = sympify(q_expr)
        r_sym = sympify(r_expr)

        h = (b - a) / n
        # Nodos interiores: x_1, x_2, ..., x_{n-1}
        nodos = [a + i * h for i in range(n + 1)]
        interiores = nodos[1:-1]  # n-1 nodos
        m = len(interiores)

        pasos = []
        pasos.append(f"EDO: y'' + p(x)·y' + q(x)·y = r(x)")
        pasos.append(f"p(x) = {p_expr}  |  q(x) = {q_expr}  |  r(x) = {r_expr}")
        pasos.append(f"Intervalo: [{a}, {b}]  |  n = {n}  →  h = {round(h, 8)}")
        pasos.append(f"Condiciones: y({a}) = {ya},  y({b}) = {yb}")
        pasos.append(f"Nodos interiores: {m}")
        pasos.append(f"\nDiscretización (diferencias centradas):")
        pasos.append(f"  y'' ≈ (y_{{i-1}} - 2y_i + y_{{i+1}}) / h²")
        pasos.append(f"  y'  ≈ (y_{{i+1}} - y_{{i-1}}) / (2h)")
        pasos.append(f"\nSustituyendo:")
        pasos.append(f"  (1/h² - p_i/2h)·y_{{i-1}} + (q_i - 2/h²)·y_i + (1/h² + p_i/2h)·y_{{i+1}} = r_i")

        # Construir sistema tridiagonal A·y = b_vec
        A_mat = [[0.0] * m for _ in range(m)]
        b_vec = [0.0] * m

        pasos.append(f"\nPaso 1: Construir sistema tridiagonal {m}×{m}")

        for i, xi in enumerate(interiores):
            pi = float(p_sym.subs(x, xi).evalf())
            qi = float(q_sym.subs(x, xi).evalf())
            ri = float(r_sym.subs(x, xi).evalf())

            sub  = 1/h**2 - pi/(2*h)   # coef y_{i-1}
            diag = qi - 2/h**2          # coef y_i
            sup  = 1/h**2 + pi/(2*h)   # coef y_{i+1}

            A_mat[i][i] = diag
            if i > 0:
                A_mat[i][i-1] = sub
            if i < m - 1:
                A_mat[i][i+1] = sup

            rhs = ri
            # Aplicar condiciones de frontera
            if i == 0:
                rhs -= sub * ya
            if i == m - 1:
                rhs -= sup * yb

            b_vec[i] = rhs
            pasos.append(f"  i={i+1}, x={round(xi,6)}: sub={round(sub,6)}, diag={round(diag,6)}, sup={round(sup,6)}, rhs={round(rhs,6)}")

        # Resolver con eliminación gaussiana (tridiagonal)
        pasos.append(f"\nPaso 2: Resolver sistema por eliminación gaussiana")
        A_np = np.array(A_mat, dtype=float)
        b_np = np.array(b_vec, dtype=float)

        try:
            sol = np.linalg.solve(A_np, b_np).tolist()
        except np.linalg.LinAlgError:
            return {"success": False, "error": "Sistema singular — revisa los coeficientes"}

        # Solución completa incluyendo condiciones de frontera
        y_sol = [ya] + sol + [yb]

        pasos.append(f"\nPaso 3: Solución en cada nodo")
        tabla = []
        for i, (xi, yi) in enumerate(zip(nodos, y_sol)):
            pasos.append(f"  y({round(xi, 6)}) = {round(yi, 8)}")
            tabla.append({"i": i, "x": round(xi, 8), "y": round(yi, 8)})

        return {
            "success": True,
            "h": h,
            "n_interiores": m,
            "pasos": pasos,
            "tabla": tabla,
            "x_plot": [round(v, 8) for v in nodos],
            "y_plot": [round(v, 8) for v in y_sol],
        }

    except Exception as e:
        return {"success": False, "error": str(e)}