#!/usr/bin/env python3
"""Script para insertar estilos CSS y JavaScript del procedimiento XGBoost"""

# Leer el archivo
with open('app/routers/detailed_analysis.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Verificar si ya está insertado
if '.xgboost-btn' in content:
    print("⚠️  Los estilos ya están presentes en el archivo")
    exit(0)

# Buscar donde insertar (después de @keyframes spin)
marker = """        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}"""

if marker not in content:
    print("❌ No se encontró el marcador")
    exit(1)

# CSS a agregar (continuará en siguiente mensaje por límite de espacio)
css_addition = """

        /* ===== ESTILOS PARA EL PROCEDIMIENTO XGBOOST ANIMADO ===== */
        .xgboost-btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 16px 32px;
            font-size: 1.1rem;
            border-radius: 12px;
            cursor: pointer;
            margin-top: 24px;
            transition: all 0.3s ease;
            font-weight: 600;
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        }}

        .xgboost-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 24px rgba(102, 126, 234, 0.4);
        }}

        .xgboost-process-container {{
            background: #141414;
            border-radius: 16px;
            padding: 40px;
            margin: 30px 0;
            border: 1px solid #2a2a2a;
        }}

        .process-header {{
            text-align: center;
            margin-bottom: 50px;
        }}

        .process-header h2 {{
            font-size: 2.2rem;
            color: #ffffff;
            margin-bottom: 10px;
        }}

        .process-header p {{
            color: #a0a0a0;
            font-size: 1.1rem;
        }}

        .animated-step {{
            background: #1a1a1a;
            border-radius: 12px;
            padding: 30px;
            margin: 30px 0;
            border: 1px solid #2a2a2a;
            animation: fadeInUp 0.6s ease-out;
        }}

        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(30px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .step-visual {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 40px;
            background: #0d0d0d;
            border-radius: 12px;
            margin-bottom: 20px;
            min-height: 180px;
            flex-wrap: wrap;
            gap: 20px;
        }}

        .step-explanation {{ padding: 20px; }}
        .step-explanation h3 {{ color: #667eea; font-size: 1.5rem; margin-bottom: 12px; }}
        .step-explanation p {{ color: #d0d0d0; font-size: 1.05rem; line-height: 1.7; }}

        .data-input {{ display: flex; gap: 12px; }}
        .data-point {{
            width: 30px;
            height: 30px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); opacity: 1; }}
            50% {{ transform: scale(1.3); opacity: 0.6; }}
        }}

        .arrow-flow {{
            font-size: 2.5rem;
            color: #667eea;
            margin: 0 20px;
            animation: slideRight 1.5s ease-in-out infinite;
        }}

        @keyframes slideRight {{
            0%, 100% {{ transform: translateX(0); }}
            50% {{ transform: translateX(10px); }}
        }}

        .process-box {{
            background: #1e1e1e;
            padding: 25px 35px;
            border-radius: 12px;
            border: 2px solid #667eea;
            text-align: center;
            color: #ffffff;
            font-weight: 600;
        }}

        .process-box i {{ font-size: 2.5rem; color: #667eea; display: block; margin-bottom: 10px; }}
        .formula-box {{ background: #0d0d0d; padding: 30px; border-radius: 12px; border: 2px solid #667eea; }}
        .formula-line {{ display: flex; align-items: center; gap: 15px; margin: 15px 0; font-size: 1.3rem; }}
        .formula-label {{ color: #a0a0a0; font-weight: 500; }}
        .formula-value {{ color: #ffffff; font-family: 'Courier New', monospace; font-size: 1.4rem; font-weight: 600; }}

        .trees-container {{ display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 15px; }}
        .tree-item {{ display: flex; flex-direction: column; align-items: center; animation: treeGrow 1s ease-out; }}

        @keyframes treeGrow {{
            from {{ transform: scale(0) rotate(-180deg); opacity: 0; }}
            to {{ transform: scale(1) rotate(0); opacity: 1; }}
        }}

        .tree-icon {{ font-size: 3rem; animation: sway 3s ease-in-out infinite; }}

        @keyframes sway {{
            0%, 100% {{ transform: rotate(-5deg); }}
            50% {{ transform: rotate(5deg); }}
        }}

        .tree-label {{ color: #ffffff; font-weight: 600; margin-top: 8px; }}
        .tree-weight {{ color: #667eea; font-size: 1.2rem; font-family: 'Courier New', monospace; }}
        .plus-symbol {{ font-size: 2rem; color: #667eea; font-weight: bold; }}
        .ellipsis {{ font-size: 2rem; color: #a0a0a0; letter-spacing: 5px; }}

        .gradient-visual {{ display: flex; flex-direction: column; align-items: center; gap: 20px; }}
        .gradient-curve {{ background: #0d0d0d; border-radius: 12px; padding: 20px; }}
        .gradient-label {{ color: #ffffff; font-size: 1.2rem; display: flex; align-items: center; gap: 10px; }}
        .gradient-label i {{ color: #f5576c; animation: bounce 1.5s ease-in-out infinite; }}

        @keyframes bounce {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-10px); }}
        }}

        .regularization-demo {{ display: flex; align-items: center; gap: 30px; flex-wrap: wrap; justify-content: center; }}
        .overfitting-tree, .balanced-tree {{ text-align: center; padding: 20px; border-radius: 12px; min-width: 150px; }}
        .overfitting-tree {{ background: rgba(239, 68, 68, 0.1); border: 2px solid #ef4444; }}
        .balanced-tree {{ background: rgba(34, 197, 94, 0.1); border: 2px solid #22c55e; }}
        .tree-complex, .tree-simple {{ font-size: 1.5rem; margin-bottom: 10px; }}
        .complexity-indicator {{ color: #a0a0a0; font-size: 0.9rem; }}
        .arrow-transform {{ font-size: 1.5rem; color: #667eea; font-weight: bold; }}

        .prediction-flow {{ display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 20px; }}
        .input-box, .ensemble-box, .voting-box, .output-box {{
            background: #1e1e1e;
            padding: 20px 30px;
            border-radius: 12px;
            border: 2px solid #667eea;
            text-align: center;
            color: #ffffff;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }}

        .input-box i, .voting-box i, .output-box i {{ font-size: 2.5rem; color: #667eea; }}
        .ensemble-box {{ display: flex; flex-direction: column; align-items: center; }}
        .mini-tree {{ font-size: 1.5rem; display: inline-block; margin: 0 5px; animation: sway 3s ease-in-out infinite; }}
        .ensemble-label {{ color: #a0a0a0; font-size: 0.9rem; margin-top: 8px; }}
        .pulsing {{ animation: pulseGlow 2s ease-in-out infinite; }}

        @keyframes pulseGlow {{
            0%, 100% {{ box-shadow: 0 0 10px #667eea; border-color: #667eea; }}
            50% {{ box-shadow: 0 0 30px #667eea, 0 0 50px #764ba2; border-color: #764ba2; }}
        }}

        .mini-formula {{
            background: #0d0d0d;
            padding: 15px 20px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            color: #ffffff;
            font-size: 1.2rem;
            margin-top: 15px;
            border: 1px solid #667eea;
            text-align: center;
        }}

        .process-footer {{ text-align: center; margin-top: 50px; }}
        .close-process-btn {{
            background: #2a2a2a;
            color: #ffffff;
            border: 1px solid #3a3a3a;
            padding: 12px 32px;
            font-size: 1rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        .close-process-btn:hover {{ background: #3a3a3a; transform: translateY(-2px); }}

        @media (max-width: 768px) {{
            .step-visual {{ flex-direction: column; }}
            .trees-container {{ flex-direction: column; }}
            .prediction-flow {{ flex-direction: column; }}
            .arrow-flow {{ transform: rotate(90deg); }}
        }}
    </style>

    <script>
        function toggleXGBoostProcess() {{
            const process = document.getElementById('xgboostProcess');
            const btn = document.getElementById('xgboostBtn');
            if (process.style.display === 'none' || process.style.display === '') {{
                process.style.display = 'block';
                btn.innerHTML = '<i class="fas fa-eye-slash"></i> Ocultar Procedimiento';
                process.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }} else {{
                process.style.display = 'none';
                btn.innerHTML = '<i class="fas fa-play-circle"></i> Ver Procedimiento XGBoost Animado';
            }}
        }}
    </script>
</body>
</html>\""""

# Reemplazar
new_content = content.replace(marker + "\n        }}", marker + "}}" + css_addition)

# Guardar
with open('app/routers/detailed_analysis.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ Archivo actualizado correctamente!")
print("✅ CSS y JavaScript agregados para el procedimiento XGBoost animado")
