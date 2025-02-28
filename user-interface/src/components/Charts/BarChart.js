import React, { Component } from 'react';
import Chart from 'react-apexcharts';

class BarChart extends Component {
  constructor(props) {
    super(props);
    this.state = {
      chartData: [],
      chartOptions: {},
    };
  }

  componentDidMount() {
    const { barChartData, barChartOptions } = this.props;

    this.setState({
      chartData: barChartData,
      chartOptions: barChartOptions,
    });
  }

  componentDidUpdate(prevProps) {
    if (prevProps.barChartData !== this.props.barChartData) {
      this.setState({ chartData: this.props.barChartData });
    }
    if (prevProps.barChartOptions !== this.props.barChartOptions) {
      this.setState({ chartOptions: this.props.barChartOptions });
    }
  }

  render() {
    return (
      <Chart
        options={this.state.chartOptions}
        series={this.state.chartData}
        height="350px"
        type="bar"
      />
    );
  }
}

export default BarChart;
