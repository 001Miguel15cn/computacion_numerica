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
