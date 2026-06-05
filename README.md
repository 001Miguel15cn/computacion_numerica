# computacion_numerica
### Configuracion basica de ambiente

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

pip install uvicorn

### lanzar backend

uvicorn backend.main:app --reload

### lanzar frontend en VS code

en el archivo index.html hacer click derecho y clickear en Open With Life Server
