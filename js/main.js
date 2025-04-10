// Configuración de los gráficos usando Chart.js
window.onload = function () {
  // Gráfico de potencia de inversores
  const powerCtx = document.getElementById("powerChart").getContext("2d");
  const powerChart = new Chart(powerCtx, {
    type: "line",
    data: {
      labels: [
        "7:00",
        "8:00",
        "9:00",
        "10:00",
        "11:00",
        "12:00",
        "13:00",
        "14:00",
        "15:00",
        "16:00",
        "17:00",
        "18:00",
      ],
      datasets: [
        {
          label: "Inversor 01",
          data: [0, 3.2, 8.5, 10.2, 11.8, 12.1, 12.0, 11.5, 10.2, 7.8, 3.5, 0],
          borderColor: "#4CAF50",
          tension: 0.4,
          fill: false,
        },
        {
          label: "Inversor 02",
          data: [0, 3.4, 8.7, 10.5, 12.1, 12.3, 12.2, 11.8, 10.5, 8.0, 3.7, 0],
          borderColor: "#2196F3",
          tension: 0.4,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true, // Mantiene las proporciones
      plugins: {
        legend: {
          position: "bottom",
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Potencia (kW)",
          },
        },
      },
    },
  });

  // Gráfico de condiciones ambientales
  const envCtx = document.getElementById("environmentChart").getContext("2d");
  const envChart = new Chart(envCtx, {
    type: "line",
    data: {
      labels: [
        "00:00",
        "01:00",
        "02:00",
        "03:00",
        "04:00",
        "05:00",
        "06:00",
        "07:00",
        "08:00",
        "09:00",
        "10:00",
        "11:00",
        "12:00",
      ],
      datasets: [
        {
          label: "Temperatura (°C)",
          data: [
            37.0, 36.2, 35.8, 35.5, 35.0, 34.8, 35.2, 36.5, 37.8, 38.5, 38.8,
            39.0, 39.2,
          ],
          borderColor: "#F44336",
          tension: 0.4,
          fill: false,
          yAxisID: "y",
        },
        {
          label: "Velocidad del viento (m/s)",
          data: [
            1.2, 1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 2.6, 2.4, 2.2, 2.0, 1.8, 1.5,
          ],
          borderColor: "#2196F3",
          tension: 0.4,
          fill: false,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true, // Evita que la gráfica se expanda demasiado
      plugins: {
        legend: {
          position: "bottom",
        },
      },
      scales: {
        y: {
          type: "linear",
          display: true,
          position: "left",
          title: {
            display: true,
            text: "Temperatura (°C)",
          },
        },
        y1: {
          type: "linear",
          display: true,
          position: "right",
          title: {
            display: true,
            text: "Velocidad (m/s)",
          },
          grid: {
            drawOnChartArea: false,
          },
        },
      },
    },
  });

  // Gráfico de correlación entre irradiancia y producción
  const correlationCtx = document
    .getElementById("correlationChart")
    .getContext("2d");
  const correlationChart = new Chart(correlationCtx, {
    type: "line",
    data: {
      labels: [
        "07:00",
        "08:00",
        "09:00",
        "10:00",
        "11:00",
        "12:00",
        "13:00",
        "14:00",
        "15:00",
        "16:00",
        "17:00",
        "18:00",
      ],
      datasets: [
        {
          label: "Irradiancia Solar (W/m²)",
          data: [0, 125, 267.5, 450, 650, 750, 800, 750, 650, 450, 125, 0],
          borderColor: "#FFC107",
          tension: 0.4,
          fill: false,
          yAxisID: "y",
        },
        {
          label: "Producción Total AC (kW)",
          data: [0, 50, 120, 180, 220, 247.8, 245, 220, 180, 120, 50, 0],
          borderColor: "#4CAF50",
          tension: 0.4,
          fill: false,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          position: "bottom",
        },
        title: {
          display: true,
          text: "Correlación Diaria",
        },
      },
      scales: {
        y: {
          type: "linear",
          display: true,
          position: "left",
          title: {
            display: true,
            text: "Irradiancia (W/m²)",
          },
        },
        y1: {
          type: "linear",
          display: true,
          position: "right",
          title: {
            display: true,
            text: "Potencia AC (kW)",
          },
          grid: {
            drawOnChartArea: false,
          },
        },
      },
    },
  });
};

// Función para cambiar entre pestañas
function openTab(evt, tabName) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  tablinks = document.getElementsByClassName("tablink");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  document.getElementById(tabName).style.display = "block";
  evt.currentTarget.className += " active";
}

// Mostrar la primera pestaña por defecto
document.addEventListener("DOMContentLoaded", function () {
  document.querySelector(".tablink").click();
});
