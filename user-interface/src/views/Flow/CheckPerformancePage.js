import React, { useEffect, useMemo, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useParams, useHistory } from "react-router-dom";
import {
  Box,
  Flex,
  Grid,
  IconButton,
  Divider,
  Text,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatGroup,
  CircularProgress,
  CircularProgressLabel,
} from "@chakra-ui/react";
import Card from "components/Card/Card";
import CardBody from "components/Card/CardBody";
import CardHeader from "components/Card/CardHeader";
import Chart from "react-apexcharts";
import { ArrowForwardIcon, ArrowBackIcon } from "@chakra-ui/icons";
import {
  fetchSurrogateFeatureImportance,
  fetchSurrogateMatric,
  fetchSurrogateResult,
} from "store/features/flowSlice";
import { fetchFlowProperties } from "store/features/flowSlice";

const SurrogatePerformancePage = () => {
  const { flowId, projectId } = useParams();
  const dispatch = useDispatch();
  const history = useHistory();

  const [outputIndex, setOutputIndex] = useState(0);

  // Redux: surrogate data
  const featureImportance = useSelector(
    (state) => state.flows.surrogateFeatureImportance[flowId] || []
  );
  const surrogateMatric = useSelector(
    (state) => state.flows.surrogateMatric[flowId] || []
  );
  const surrogateResult = useSelector(
    (state) => state.flows.surrogateResult[flowId] || []
  );

  const redux = useSelector((state) => state);

  console.log(redux);

  useEffect(async () => {
    await dispatch(fetchSurrogateFeatureImportance(flowId)).unwrap();
    await dispatch(fetchSurrogateMatric(flowId)).unwrap();
    await dispatch(fetchSurrogateResult(flowId)).unwrap();
  }, [dispatch, flowId]); // properties가 아무것도 없을 때만 fetch

  const currentColumnType = surrogateMatric[outputIndex]?.column_type;
  console.log(currentColumnType);

  const metricsCombinedCard = (
    <Card mb={4} h="100%" minH="400px">
      <CardHeader pb={2}>
        <Flex w="100%" justifyContent="space-between" alignItems="center">
          {/* 왼쪽: 제목 및 컬럼 이름 */}
          <Box>
            <Text fontSize="xl" fontWeight="bold">
              Metrics
            </Text>
            <Text color="gray.400">
              {surrogateMatric[outputIndex]?.column_name}
            </Text>
          </Box>
          {/* 오른쪽: 네비게이션 버튼 */}
          {surrogateMatric.length > 1 && (
            <Box>
              <Flex alignItems="center">
                <IconButton
                  icon={<ArrowBackIcon />}
                  onClick={() =>
                    setOutputIndex((prev) => Math.max(prev - 1, 0))
                  }
                  isDisabled={outputIndex === 0}
                  size="sm"
                  mr={2}
                  aria-label="Previous Output"
                />
                <Text fontSize="sm" color="gray.400">
                  Output {outputIndex + 1} of {surrogateMatric.length}
                </Text>
                <IconButton
                  icon={<ArrowForwardIcon />}
                  onClick={() =>
                    setOutputIndex((prev) =>
                      Math.min(prev + 1, surrogateMatric.length - 1)
                    )
                  }
                  isDisabled={outputIndex === surrogateMatric.length - 1}
                  size="sm"
                  ml={2}
                  aria-label="Next Output"
                />
              </Flex>
            </Box>
          )}
        </Flex>
      </CardHeader>

      <CardBody>
        <Flex
          direction="column"
          align="center"
          justify="center"
          gap={10}
          w="100%"
        >
          {currentColumnType === "numerical" && (
            // numeric인 경우 RMSE, MAE, Lower is better 섹션 렌더링
            <Flex direction="column" align="center" mt="4" w="100%">
              <Flex direction="row" justifyContent="center" gap={4} w="100%">
                <Card align="center">
                  <StatGroup justifyContent="center">
                    <Stat>
                      <StatLabel>
                        <Text color="gray.400" fontSize="sm">
                          RMSE
                        </Text>
                      </StatLabel>
                      <StatNumber fontSize={{ md: "36px", lg: "42px" }}>
                        {surrogateMatric[outputIndex]?.rmse.toFixed(4)}
                      </StatNumber>
                    </Stat>
                  </StatGroup>
                </Card>
                <Card align="center">
                  <StatGroup justifyContent="center">
                    <Stat>
                      <StatLabel>
                        <Text color="gray.400" fontSize="sm">
                          MAE
                        </Text>
                      </StatLabel>
                      <StatNumber fontSize={{ md: "36px", lg: "42px" }}>
                        {surrogateMatric[outputIndex]?.mae.toFixed(4)}
                      </StatNumber>
                    </Stat>
                  </StatGroup>
                </Card>
              </Flex>
              <Box mt={4} w="100%">
                <Card
                  borderRadius="md"
                  boxShadow="sm"
                  p={2}
                  bg="rgba(20, 20, 232, 0.2)"
                  mx="auto"
                  w={{ base: "90%", md: "60%" }}
                >
                  <Text fontSize="sm" color="#fff" textAlign="center">
                    Lower is better
                  </Text>
                </Card>
              </Box>
            </Flex>
          )}

          {/* R-squared 섹션 (numeric, categorical 모두 공통으로 렌더링) */}
          <Flex direction="column" align="center" w="100%">
            <CircularProgress
              max="1"
              min="-1"
              value={Number(surrogateMatric[outputIndex]?.r_squared.toFixed(3))}
              size={
                window.innerWidth >= 1024
                  ? 200
                  : window.innerWidth >= 768
                  ? 170
                  : 200
              }
              thickness={9}
              color="rgba(5,205,153,0.4)"
            >
              <CircularProgressLabel>
                <Flex direction="column" justify="center" align="center">
                  <Text color="gray.400" fontSize="sm">
                    {currentColumnType === "numerical"
                      ? "R-Squared"
                      : "Accuracy"}
                  </Text>
                  <Text
                    color="#fff"
                    fontSize={{ md: "36px", lg: "50px" }}
                    fontWeight="bold"
                    mb="4px"
                  >
                    {surrogateMatric[outputIndex]?.r_squared.toFixed(3)}
                  </Text>
                </Flex>
              </CircularProgressLabel>
            </CircularProgress>
            <Box mt={4} w="100%">
              <Card
                borderRadius="md"
                boxShadow="sm"
                p={2}
                bg="rgba(5,205,153,0.2)"
                mx="auto"
                w={{ base: "90%", md: "60%" }}
              >
                <Text fontSize="sm" color="#fff" textAlign="center">
                  Higher is better
                </Text>
              </Card>
            </Box>
          </Flex>
        </Flex>
      </CardBody>
    </Card>
  );

  // Feature Importance: 정렬 후 수평 바 차트 (백분율 표시)
  const sortedImportance = useMemo(() => {
    return [...featureImportance].sort((a, b) => b.importance - a.importance);
  }, [featureImportance]);

  const importanceSeries = useMemo(() => {
    return [
      {
        name: "Importance (%)",
        data: sortedImportance.map((fi) =>
          Number((fi.importance * 100).toFixed(2))
        ),
      },
    ];
  }, [sortedImportance]);

  const importanceOptions = useMemo(() => {
    return {
      chart: { type: "bar", toolbar: { show: false } },
      plotOptions: {
        bar: {
          borderRadius: 8,
          columnWidth: "12px",
        },
      },
      dataLabels: {
        enabled: false,
      },
      grid: {
        show: false,
      },
      fill: {
        colors: "#fff",
      },
      xaxis: {
        title: { text: "Importance (%)", style: { color: "#fff" } },
        labels: {
          show: true,
          style: {
            colors: "rgba(211, 211, 211, 0.71)",
            fontSize: "12px",
          },
        },
        axisBorder: {
          show: false,
        },
        axisTicks: {
          show: false,
        },
        categories: sortedImportance.map((fi) => fi.column_name),
      },
      yaxis: {
        show: true,
        color: "#fff",
        categories: sortedImportance.map((fi) => fi.column),
        labels: {
          show: true,
          style: {
            colors: "#fff",
            fontSize: "12px",
          },
        },
      },
      tooltip: {
        theme: "dark",
        y: { formatter: (val) => `${val}%` },
        style: {
          fontSize: "14px",
        },
        onDatasetHover: {
          style: {
            fontSize: "14px",
          },
        },
      },
      responsive: [
        {
          breakpoint: 768,
          options: {
            plotOptions: {
              bar: {
                borderRadius: 0,
              },
            },
          },
        },
      ],
      // 바 색상는 그대로 사용
    };
  }, [sortedImportance]);

  const importanceCard = (
    <Card h="100%" minH="350px">
      <CardHeader pb={2}>
        <Text fontSize="xl" fontWeight="bold">
          Feature Importance
        </Text>
      </CardHeader>
      <Box w="100%" h="100%" mt={2}>
        {sortedImportance.length > 0 ? (
          <Chart
            options={importanceOptions}
            series={importanceSeries}
            type="bar"
            height="100%"
          />
        ) : (
          <Text>No feature importance data available</Text>
        )}
      </Box>
    </Card>
  );

  /*** 두 번째 탭: Best & Worst Cases (Line Chart) ***/
  // 정렬: rank 기준 (낮을수록 좋은 것으로 가정)
  const sortedResults = useMemo(() => {
    return surrogateResult.slice().sort((a, b) => a.rank - b.rank);
  }, [surrogateResult]);

  const bestCases = useMemo(() => sortedResults.slice(0, 5), [sortedResults]);
  const worstCases = useMemo(() => sortedResults.slice(-5), [sortedResults]);
  const numericOutputs = useMemo(
    () =>
      surrogateMatric.filter((output) => output.column_type === "numerical"),
    [surrogateMatric]
  );

  // 각 케이스에 대해 "Ground Truth"와 "Predicted"를 선 차트로 표현
  const bestCasesLineData = useMemo(() => {
    return {
      // x축 레이블 제거
      categories: [],
      series: [
        {
          name: "Ground Truth",
          data: bestCases.map((res) => res.ground_truth),
        },
        { name: "Predicted", data: bestCases.map((res) => res.predicted) },
      ],
    };
  }, [bestCases]);

  const worstCasesLineData = useMemo(() => {
    return {
      categories: [],
      series: [
        {
          name: "Ground Truth",
          data: worstCases.map((res) => res.ground_truth),
        },
        { name: "Predicted", data: worstCases.map((res) => res.predicted) },
      ],
    };
  }, [worstCases]);

  // 라인 차트 옵션 수정: x축 레이블 제거, 데이터 라벨 색상 변경
  const lineChartOptions = useMemo(() => {
    return {
      chart: { type: "line", toolbar: { show: false } },
      dataLabels: {
        enabled: false,
      },
      stroke: { curve: "smooth" },
      xaxis: {
        labels: { colors: "#c8cfca", fontSize: "12px" },
        axisBorder: { show: false },
      },
      yaxis: {
        labels: {
          formatter: (val) => val.toFixed(3),
          style: {
            colors: "#fff",
            fontSize: "12px",
            fontFamily: "Plus Jakarta Display",
          },
        },
      },
      tooltip: {
        theme: "dark",
        style: {
          fontSize: "14px",
          fontFamily: "Plus Jakarta Display",
        },
      },
      legend: {
        show: true,
        labels: {
          colors: "#fff", // 원하는 색상 코드 지정
        },
      },
      colors: ["#2CD9FF", "#582CFF"],
      grid: {
        padding: {
          right: 200,
        },
      },
      axisBorder: {
        show: false,
      },
      axisTicks: {
        show: false,
      },
      fill: {
        type: "gradient",
        gradient: {
          shade: "light",
          type: "vertical",
          shadeIntensity: 0.5,
          gradientToColors: ["#2CD9FF", "#582CFF"], // optional, if not defined - uses the shades of same color in series
          inverseColors: false,
          opacityFrom: 0.5,
          opacityTo: 0.1,
          stops: [0, 90, 100],
        },
      },
      grid: {
        strokeDashArray: 5,
        borderColor: "#56577A",
      },
    };
  }, []);

  const bestCasesCard = (
    <Card minH="300px" mb={4}>
      <CardHeader pb={2}>
        <Text fontSize="xl" fontWeight="bold">
          Best Cases
        </Text>
      </CardHeader>

      <Box w="100%" minH={{ sm: "500px" }}>
        {bestCases.length > 0 ? (
          <Chart
            options={lineChartOptions}
            series={bestCasesLineData.series}
            type="area"
            height="100%"
          />
        ) : (
          <Text>No best cases available</Text>
        )}
      </Box>
    </Card>
  );

  const worstCasesCard = (
    <Card minH="300px" mb={4}>
      <CardHeader pb={2}>
        <Text fontSize="xl" fontWeight="bold">
          Worst Cases
        </Text>
      </CardHeader>

      <Box w="100%" minH={{ sm: "500px" }}>
        {worstCases.length > 0 ? (
          <Chart
            options={lineChartOptions}
            series={worstCasesLineData.series}
            type="area"
            width="100%"
            height="100%"
          />
        ) : (
          <Text>No worst cases available</Text>
        )}
      </Box>
    </Card>
  );

  return (
    <Flex
      flexDirection="column"
      pt={{ base: "120px", md: "75px" }}
      px={4}
      maxH="100vh"
      color="white"
    >
      {/* 헤더 영역 */}
      <Flex justifyContent="space-between" alignItems="center" mb={6}>
        <IconButton
          icon={<ArrowBackIcon />}
          onClick={() => history.goBack()}
          colorScheme="blue"
        />
        <Box textAlign="center">
          <Text fontSize="2xl" fontWeight="bold">
            Surrogate Model Performance
          </Text>
          <Text fontSize="sm" color="gray.400">
            Check metrics, feature importance, and prediction cases.
          </Text>
        </Box>
        <IconButton
          icon={<ArrowForwardIcon />}
          onClick={() =>
            history.push(
              `/projects/${projectId}/flows/${flowId}/optimization-results`
            )
          }
          colorScheme="blue"
        />
      </Flex>

      {/* 탭 영역 */}
      <Tabs variant="enclosed" colorScheme="red">
        <TabList>
          <Tab>Metrics & Feature Importance</Tab>
          {numericOutputs.length > 0 && <Tab>Prediction Cases</Tab>}
        </TabList>
        <TabPanels>
          {/* 첫 번째 탭: Metrics & Feature Importance */}
          <TabPanel>
            <Grid
              templateColumns={{ base: "1fr", md: "2fr 1fr" }}
              h="calc(80vh - 150px)"
              gap={6}
            >
              {importanceCard}
              {metricsCombinedCard}
            </Grid>
          </TabPanel>

          {/* 두 번째 탭: Prediction Cases */}
          {numericOutputs.length > 0 && (
            <TabPanel>
              {surrogateMatric[outputIndex]?.column_type === "numerical" ? (
                <Grid
                  templateColumns={{ base: "1fr", md: "1fr 1fr" }}
                  h="calc(80vh - 150px)"
                  gap={6}
                >
                  {bestCasesCard}
                  {worstCasesCard}
                </Grid>
              ) : (
                <Flex align="center" justify="center" h="calc(80vh - 150px)">
                  <Text fontSize="xl" color="gray.400">
                    Prediction cases are available only for numerical outputs.
                  </Text>
                </Flex>
              )}
            </TabPanel>
          )}
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default SurrogatePerformancePage;
