import React from 'react';
import Chart from 'react-apexcharts';

function PieChart({ data, colorsArray }) {
  // data: [{ label: 'Numeric', value: 10 }, { label: 'Categorical', value: 5 }, ...]
  const series = data.map(item => item.value);
  const options = {
    chart: {
      type: 'pie',
    },
    labels: data.map(item => item.label),
    stroke: {
      show: false,
    },
    legend: {
      show: false,
    },
    dataLabels: {
      enabled: true, // 각 영역에 라벨 보이게 함
      formatter: function (val, opts) {
        // opts.seriesIndex를 이용해서 labels 배열에서 라벨을 가져올 수 있음.
        // 예시로 라벨과 값(혹은 퍼센트)를 표시하는 formatter:
        const label = opts.w.config.labels[opts.seriesIndex];
        // val은 퍼센트 값이 전달됨.
        return `${label}: ${val.toFixed(1)}%`;
      },
      style: {
        fontSize: '10px',
        colors: ['#fff'],
      },
    },
    colors: colorsArray || [
      'rgba(207, 158, 239, 1)',
      'rgba(199, 163, 239, 0.9)',
      'rgba(191, 169, 239, 0.8)',
      'rgba(184, 174, 239, 0.7)',
      'rgba(176, 180, 239, 0.6)',
      'rgba(168, 185, 240, 0.5)',
      'rgba(160, 191, 240, 0.6)',
      'rgba(153, 196, 240, 0.7)',
      'rgba(145, 202, 240, 0.8)',
      'rgba(137, 207, 240, 0.9)',
    ],

    tooltip: {
      theme: 'dark',
    },
  };

  return <Chart options={options} series={series} type="pie" height="100%" />;
}

export default PieChart;
