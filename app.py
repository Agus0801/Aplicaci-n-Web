import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from auxiliar import dias_meses, nombres_meses

#Configuracion de la parte lateral de la app
st.set_page_config(
    page_title="Datos Climatológicos 2019",
    layout="wide",
    initial_sidebar_state="collapsed"  
)

with st.sidebar:
  with st.expander("Seleccion de fecha"):
   nombre_mes = st.selectbox("Mes", nombres_meses)
   mes = nombres_meses.index(nombre_mes) + 1
   dias = st.date_input("Seleccionar día",
                       value=datetime.date(2019, 1, 1),
                       min_value=datetime.date(2019, 1, 1),
                       max_value=datetime.date(2019, 12, 31))

  with st.expander("Opciones de los paneles e inversor"):
    st.subheader("Paneles:")
    N = st.number_input('Número de paneles',
                        value=12,
                        min_value=0,
                        max_value=1000,
                        step=1)
    Ppico = st.number_input('Potencia pico del panel [Watts]',
                            value=240,
                            min_value=0,
                            max_value=1000,
                            step=1)
    kp = st.number_input('Coef. de potencia-temperatura [1/°C]',
                         value=-0.0044,
                         min_value=-0.1,
                         max_value=0.0,
                         step=0.0001,
                         format="%.4f")
    rend = st.number_input('Rendimiento [%]',
                           min_value=0.,
                           max_value=1.,
                           value=0.9,
                           step=0.01)

    Gstd = st.number_input('Irradiancia Estandar [W/m^2]',
                           value=1000,
                           min_value=0,
                           max_value=2000,
                           step=1)

    Tr = st.number_input('Temperatura de referencia[°C]',
                         min_value=-273.,
                         max_value=1000.,
                         value=25.,
                         step=0.01)
    st.subheader("Opciones del inversor:")
    mu = st.number_input('Umbral min. [%]',
                         value=0.010,
                         min_value=0.,
                         max_value=1.,
                         step=0.001)
    Pinv = st.number_input('Potencia inversor [Kw]',
                           value=2.5,
                           min_value=0.,
                           step=0.1)

    Pmin = (mu / 100) * Pinv

st.title("Generador Fotovoltaico - Santa Fe")

st.markdown(
    "<hr style='border: 0; height: 4px; background: linear-gradient(to right, yellow, green, blue);'>",
    unsafe_allow_html=True,
)

st.markdown("## Links que te pueden interesar")

col1, col2, col3 = st.columns(3)
with col1: 
    st.markdown(
        """
        <a href="https://www.epe.santafe.gov.ar/institucional/" target="_blank">
            <button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer;">
                Haz clic aquí para ir a la pagina de la EPE
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <a href="https://www.frsf.utn.edu.ar/noticias/606-energia-limpia-y-ahorro-energetico-un-compromiso-para-la-utn-santa-fe" target="_blank">
            <button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer;">
                Haz clic aquí para ir a la pagina de la UTN
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        """
        <a href="https://www.santafe.gob.ar/ms/generfe/" target="_blank">
            <button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer;">
                Haz clic aquí para ir a la pagina de GENERFE
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )
st.markdown(
    "<hr style='border: 0; height: 4px; background: linear-gradient(to right, green, pink, blue);'>",
    unsafe_allow_html=True,
)
st.markdown("## Ecuaciones")
# Agrega texto explicativo y ecuaciones según sea necesario


# Potencia eléctrica P (en kilo-Watt) obtenida por un GFV:
with st.expander("Potencia eléctrica obtenida por un Generador FV (kW)"):
  st.latex(r'P [kW] = N \cdot \frac{G}{G_{\text{std}}} \cdot P_{\text{pico}}' +
         r'\cdot \left[1 + k_p \cdot (T_c - T_r)\right] \eta \cdot 10^{-3}')

# La temperatura de la celda difiere de la temperatura ambiente:
with st.expander(
    "Temperatura de la celda al diferir de la Temperatura Ambiente (°C)"):
  st.latex(
    r'{T}{c}=T+0,031 \left[ ^\circ C \cdot {m}{}^{2} \right / W]\cdot G')

# Limites de Generacion
with st.expander("Límites de Generación Técnicos (kW)"):
  st.latex(r'{P}{\min} [kW]= \frac{\mu (\%)}{100}\cdot {P}{inv}')

  st.latex(r'''{P}_{r} [kW]=  \left\{ \begin{array}{cl}
  0 & si \ P \leq {P}_{\min} \\
  P & si \ {P}{\min} < P \leq {P}{inv} \\
  {P}{inv} & si \ {P}{inv} < P
  \end{array} \right.''')


st.markdown(
    "<hr style='border: 0; height: 4px; background: linear-gradient(to right, blue, orange, red);'>",
    unsafe_allow_html=True,
)
#CALCULOS
st.markdown("## Calculos")
# Temperatura de la celda
T = st.number_input('Temperatura ambiente [°C]', value=25, min_value=-100, max_value=100)
Tc = T + 0.031 * Gstd  # Temperatura de la celda

# Potencia generada
P = N * (Gstd / 1000) * Ppico * (1 + kp * (Tc - Tr)) * rend * 1e-3

# Potencia real limitada por el inversor
Pr = max(0, min(P, Pinv))  

# Mostrar resultados
st.write(f"**Potencia generada por el sistema**: {P:.3f} kW")
st.write(f"**Temperatura de la celda**: {Tc:.1f} °C")
st.write(f"**Potencia real limitada**: {Pr:.3f} kW")


st.markdown(
    "<hr style='border: 0; height: 4px; background: linear-gradient(to right, blue, pink, yellow);'>",
    unsafe_allow_html=True,
)

# Extraxción de los datos del excel 
tabla = pd.read_excel("Datos_climatologicos_Santa_Fe_2019.xlsx", index_col=0)
tabla.index = pd.to_datetime(tabla.index)  # Asegurarse que las fechas son tipo datetime

# Centrar la tabla 
st.markdown("""
    <style>
    .css-1cpxqw2 { margin-left: auto; margin-right: auto; }
    </style>
    """, unsafe_allow_html=True)

st.write("## Tabla Anual 2019")
st.dataframe(tabla, use_container_width=True)

st.markdown(
    "<hr style='border: 0; height: 4px; background: linear-gradient(to right, red, blue, pink);'>",
    unsafe_allow_html=True,
)


st.markdown("## Graficos y otros datos")
# Tabs para análisis mensual y diario
tab1, tab2, tab3, tab4 = st.tabs(['Datos Mensuales', 'Datos Diarios', 'Datos Anuales','Otros graficos interesantes'])

# Tab 1: Datos mensuales
with tab1:
    mes = dias.month  # Obtener el mes seleccionado

    # Filtrar los datos para el mes seleccionado
    tabla_mes = tabla.loc[
        f'2019-{mes:02d}-01 00:00':f'2019-{mes:02d}-{dias_meses[mes-1]} 23:50', :
    ]

    # Calcular potencia
    tabla_mes['Potencia (kW)'] = (
        N * tabla_mes['Irradiancia (W/m²)'] / Gstd * Ppico *
        (1 + kp * (tabla_mes['Temperatura (°C)'] - Tr)) * rend * 1e-3
    )

    fig1 = px.line(
      tabla_mes['Potencia (kW)'],
      y='Potencia (kW)',
      title=f"Potencia generada en {nombre_mes} del 2019",
      line_shape="linear",
      color_discrete_sequence=['blue']  
      )
    st.plotly_chart(fig1)

    # Calcular potencia real limitada por el inversor
    tabla_mes['Pr (kW)'] = np.clip(tabla_mes['Potencia (kW)'], 0, Pinv)

    fig2 = px.line(
      tabla_mes['Pr (kW)'],
      y='Pr (kW)',
      title=f"Potencia limitada por el inversor en {nombre_mes} del 2019",
      line_shape="linear",
      color_discrete_sequence=['green']  
      )
    st.plotly_chart(fig2)

    #Grafico de temperatura
    fig3 = px.line(
      tabla_mes['Temperatura (°C)'],
      y='Temperatura (°C)',
      title=f"Temperatura de {nombre_mes} (°C)",
      line_shape="linear",
      color_discrete_sequence=['orange']  
      )
    st.plotly_chart(fig3)

     # Mostrar la tabla mensual 
    st.write(f"**Datos de {nombre_mes} con Potencia y Temperatura**")
    st.dataframe(tabla_mes[['Temperatura (°C)', 'Potencia (kW)', 'Pr (kW)']], use_container_width=True)
    

# Tab 2: Datos diarios
with tab2:
    st.write("**Gráfico de Temperatura Diario**")

    try:
        # Filtrar datos para el día seleccionado
        tabla_dia = tabla.loc[f'{dias.year}-{dias.month}-{dias.day}']
        st.line_chart(data=tabla_dia, y='Temperatura (°C)')
    except KeyError:
        st.error(f"No se encuentran datos para {dias}. Verifica la fecha seleccionada.")
        st.write("## Evaluación de los datos ") 
    # Selección de día 
   
    #Calcula la temp al medio dia y la media
    if isinstance(dias, datetime.date):    
      try:
        temp = tabla.at[f'{dias} 12:00', "Temperatura (°C)"]
        st.write(f"**Temperatura del {dias} al mediodía**")
        st.info(f"{temp} °C")
        
        temperaturas_dia = tabla.loc[f'{dias} 00:00':f'{dias} 23:59', "Temperatura (°C)"]
        media_dia = temperaturas_dia.mean()
        st.write(f"**Temperatura media del {dias}**")
        st.info(f"{media_dia:.1f} °C")
  
      except KeyError:
        st.error(f"No se encuentran datos para {dias} al mediodía.")


# Tab 3: Datos anuales
with tab3:
    np.random.seed(0)
    tabla = pd.DataFrame({
      'Fecha': pd.date_range('2019-01-01', '2019-12-31 23:50', freq='10T'),
      'Irradiancia (W/m²)': np.random.uniform(0, 1000, 52560),
      'Temperatura (°C)': np.random.uniform(10, 35, 52560),
    }).set_index('Fecha')

        # Calcular potencia
    tabla['Potencia (kW)'] = (
      N * tabla['Irradiancia (W/m²)'] / Gstd * Ppico *
      (1 + kp * (tabla['Temperatura (°C)'] - Tr)) * rend * 1e-3
    )

    # Calcular potencia limitada por el inversor
    tabla['Pr (kW)'] = np.clip(tabla['Potencia (kW)'], 0, Pinv)

    # Mostrar la tabla anual
    st.write("**Datos anuales de Potencia y Temperatura**")
    st.dataframe(tabla[['Temperatura (°C)', 'Potencia (kW)', 'Pr (kW)']], use_container_width=True)

    # Calcular la temperatura media anual
    temperatura_media_anual = tabla['Temperatura (°C)'].mean()
    st.write(f"**Temperatura media anual en 2019**")
    st.info(f"{temperatura_media_anual:.1f} °C")

    #Calcular potencia media anual
    potencia_media_anual = tabla['Potencia (kW)'].mean()
    st.write(f"**Potencia media anual en 2019**")
    st.info(f"{potencia_media_anual:.2f} kW")

    #Calcular potencia limitada por el inversor media anual 
    potenciar_media_anual = tabla['Potencia (kW)'].mean()
    st.write(f"**Potencia media anual limitada por el inversor en 2019**")
    st.info(f"{potenciar_media_anual:.2f} kW")

# Tab 4: Otros datos interesantes
with tab4:
    
    if not tabla_mes.empty:
        
        # Filtrar los datos para el día seleccionado
        tabla_dia = tabla_mes.loc[dias.strftime('%Y-%m-%d')]

        if not tabla_dia.empty:
            # Gráfico de dispersión
            st.write(f"### Relación Irradiancia vs Temperatura - {dias.strftime('%d/%m/%Y')}")
            fig, ax = plt.subplots()
            ax.scatter(tabla_dia['Irradiancia (W/m²)'], tabla_dia['Temperatura (°C)'], color="purple")
            ax.set_xlabel("Irradiancia (W/m²)")
            ax.set_ylabel("Temperatura (°C)")
            ax.set_title(f"Relación Irradiancia vs Temperatura - {dias.strftime('%d/%m/%Y')}")
            st.pyplot(fig)

            # Gráfico de barras (Distribución de temperatura)
            st.write(f"### Distribución de Temperatura - {dias.strftime('%d/%m/%Y')}")
            fig, ax = plt.subplots()
            ax.hist(tabla_dia['Temperatura (°C)'], bins=10, color="grey", edgecolor="black")
            ax.set_title(f"Distribución de Temperatura - {dias.strftime('%d/%m/%Y')}")
            ax.set_xlabel("Temperatura (°C)")
            ax.set_ylabel("Frecuencia")
            st.pyplot(fig)
        else:
            st.error(f"No hay datos disponibles para el día seleccionado: {dias.strftime('%d/%m/%Y')}.")
