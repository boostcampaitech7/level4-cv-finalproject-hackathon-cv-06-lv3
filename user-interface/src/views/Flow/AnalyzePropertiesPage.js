import React, { useEffect, useState, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useParams, useHistory } from 'react-router-dom';
import { useLocation } from 'react-router-dom';
import {
  Flex,
  Grid,
  Box,
  Text,
  IconButton,
  Divider,
  CircularProgress,
  CircularProgressLabel,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
} from '@chakra-ui/react';
import {
  ArrowBackIcon,
  ArrowForwardIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from '@chakra-ui/icons';

import Card from 'components/Card/Card';
import CardHeader from 'components/Card/CardHeader';
import CardBody from 'components/Card/CardBody';
import BarChart from 'components/Charts/BarChart';
import {
  fetchFlowProperties,
  fetchFlowDatasets,
  fetchPropertyHistograms,
} from 'store/features/flowSlice';
import { fetchCsvFilesByProject } from 'store/features/projectSlice';
import PieChart from 'components/Charts/PieChart';
import HorizontalBarChart from 'components/Charts/HorizontalBarChart';
import { fetchPropertyTypes } from 'store/features/flowSlice';

const PROPERTIES_PER_PAGE = 3; // 한 페이지당 3개

function AnalyzePropertiesPage() {
  const { projectId, flowId } = useParams();
  const dispatch = useDispatch();
  const history = useHistory();
  const location = useLocation();
  // 기본값을 numeric으로 지정해놓았지만, location에서 categorical 정보를 받아오면 그걸 적용
  const [tabIndex, setTabIndex] = useState(0);

  // -------------------------------
  // Redux 상태
  // -------------------------------
  const projectDatasets = useSelector(
    state => state.projects.datasets[projectId] || []
  );
  const histograms = useSelector(state => state.flows.histograms[flowId] || {});
  const properties = useSelector(
    state =>
      state.flows.properties?.[flowId] || {
        numerical: [],
        categorical: [],
        text: [],
        unavailable: [],
      }
  );
  const redux = useSelector(state => state);

  const flowDatasets = useSelector(state => {
    return state.flows.flows[flowId]?.csv;
  });

  const [selectedDatasets, setSelectedDatasets] = useState([]);

  useEffect(() => {
    if (flowDatasets?.length && projectDatasets?.length) {
      setSelectedDatasets(
        projectDatasets.filter(ds => flowDatasets.includes(ds.csvId))
      );
    }
  }, [flowDatasets, projectDatasets]);

  const totalSelectedRows = selectedDatasets.reduce(
    (acc, ds) => acc + (ds.rows || 0),
    0
  );

  const totalSelectedSize = selectedDatasets.reduce(
    (acc, ds) => acc + (ds.size || 0),
    0
  );

  const numericCount = properties.numerical?.length || 0;
  const categoricalCount = properties.categorical?.length || 0;
  const textCount = properties.text?.length || 0;
  const unavailableCount = properties.unavailable?.length || 0;

  // Numeric과 Categorical property 분리
  const numericProperties = properties.numerical;
  const categoricalProperties = properties.categorical;

  // -------------------------------
  // 로컬 state (페이지네이션)
  // -------------------------------
  const [numericPage, setNumericPage] = useState(0);
  const [categoricalPage, setCategoricalPage] = useState(0);
  const numericFetchedRef = useRef({});
  const categoricalFetchedRef = useRef({});

  // -------------------------------
  // 초기 로드
  // -------------------------------
  useEffect(() => {
    dispatch(fetchPropertyTypes(flowId));
    dispatch(fetchCsvFilesByProject(projectId));
    dispatch(fetchFlowDatasets(flowId));
  }, [dispatch, flowId, projectId]);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const tabFromQuery = params.get('tab');
    if (tabFromQuery === 'categorical') {
      setTabIndex(1);
    } else if (tabFromQuery === 'numeric') {
      setTabIndex(0);
    } else if (location.state && location.state.activeTab) {
      setTabIndex(location.state.activeTab === 'categorical' ? 1 : 0);
    }
  }, [location]);

  // -------------------------------
  // 페이지네이션
  // -------------------------------
  const numericTotalPages = Math.ceil(
    numericProperties.length / PROPERTIES_PER_PAGE
  );
  const categoricalTotalPages = Math.ceil(
    categoricalProperties.length / PROPERTIES_PER_PAGE
  );

  const numericCurrentPageProperties = numericProperties.slice(
    numericPage * PROPERTIES_PER_PAGE,
    (numericPage + 1) * PROPERTIES_PER_PAGE
  );
  const categoricalCurrentPageProperties = categoricalProperties.slice(
    categoricalPage * PROPERTIES_PER_PAGE,
    (categoricalPage + 1) * PROPERTIES_PER_PAGE
  );

  useEffect(() => {
    numericCurrentPageProperties.forEach(prop => {
      if (!histograms[prop] && !numericFetchedRef.current[prop]) {
        numericFetchedRef.current[prop] = true;
        dispatch(fetchPropertyHistograms({ flowId, column_name: prop }));
      }
    });
  }, [numericCurrentPageProperties, histograms, dispatch, flowId]);

  useEffect(() => {
    categoricalCurrentPageProperties.forEach(prop => {
      if (!histograms[prop] && !categoricalFetchedRef.current[prop]) {
        categoricalFetchedRef.current[prop] = true;
        dispatch(fetchPropertyHistograms({ flowId, column_name: prop }));
      }
    });
  }, [categoricalCurrentPageProperties, histograms, dispatch, flowId]);

  const handleNextNumericPage = () => {
    if (numericPage < numericTotalPages - 1) setNumericPage(p => p + 1);
  };
  const handlePrevNumericPage = () => {
    if (numericPage > 0) setNumericPage(p => p - 1);
  };

  const handleNextCategoricalPage = () => {
    if (categoricalPage < categoricalTotalPages - 1)
      setCategoricalPage(p => p + 1);
  };
  const handlePrevCategoricalPage = () => {
    if (categoricalPage > 0) setCategoricalPage(p => p - 1);
  };

  // -------------------------------
  // Next / Back
  // -------------------------------
  const handleNextStep = () => {
    history.push(`/projects/${projectId}/flows/${flowId}/configure-properties`);
  };
  const handleGoBack = () => {
    history.goBack();
  };

  function combineCategories(data, limit = 15) {
    if (data.length <= limit) return data;
    // 상위 (limit - 1)개를 선택하고, 나머지는 others로 합침.
    const topCategories = data.slice(0, limit - 1);
    const othersValue = data
      .slice(limit - 1)
      .reduce((acc, item) => acc + item.value, 0);
    return [...topCategories, { label: 'Others', value: othersValue }];
  }

  const DistributionCard = () => {
    return (
      <Card>
        <CardHeader>
          <Text color="#fff" fontSize="xl" fontWeight="bold">
            Data Properties Distribution
          </Text>
        </CardHeader>

        <CardBody
          h="500px"
          display="flex"
          flexDirection="column"
          alignItems="center"
          mt={10}
        >
          <Tabs
            index={tabIndex}
            onChange={index => setTabIndex(index)}
            variant="enclosed"
            colorScheme="red"
            w="100%"
          >
            <TabList>
              <Tab>Numeric</Tab>
              <Tab>Categorical</Tab>
            </TabList>
            <TabPanels>
              {/* Numeric Tab */}
              <TabPanel>
                {numericCurrentPageProperties.length > 0 ? (
                  <Grid
                    templateColumns="repeat(3, 1fr)"
                    gap={4}
                    w="100%"
                    mb={6}
                  >
                    {numericCurrentPageProperties.map(prop => {
                      const hData = histograms[prop];
                      let binEdges = [];
                      let counts = [];
                      let avgValue = 0;
                      let colorsArray = [];
                      let binMin = 0;
                      let binMax = 1;
                      let diff = 0;
                      if (hData) {
                        try {
                          const parsedBinEdges =
                            JSON.parse(hData.bin_edges) || [];
                          const parsedCounts = JSON.parse(hData.counts) || [];
                          binEdges = parsedBinEdges.map(edge =>
                            parseFloat(edge.toFixed(4))
                          );
                          counts = parsedCounts.map(val => parseFloat(val));
                          const total = counts.reduce(
                            (sum, val) => sum + val,
                            0
                          );
                          avgValue =
                            counts.length > 0 ? total / counts.length : 0;
                          colorsArray = counts.map(val =>
                            val === Math.max(...counts) ? '#582CFF' : '#2CD9FF'
                          );
                          if (binEdges.length > 1) {
                            binMin = binEdges[0];
                            binMax = binEdges[binEdges.length - 1];
                            diff = (binMax - binMin) * 0.03;
                          } else if (binEdges.length === 1) {
                            binMin = binEdges[0] - 1;
                            binMax = binEdges[0] + 1;
                            diff = 0;
                          }
                        } catch (e) {
                          console.error('Histogram parse error:', e);
                        }
                      }
                      return (
                        <Box key={prop} textAlign="center" h="100%">
                          <Text
                            color="gray.300"
                            fontSize="xl"
                            fontWeight="bold"
                            mb={2}
                          >
                            {prop}
                          </Text>
                          <BarChart
                            barChartData={[
                              {
                                name: prop,
                                data: counts.map((count, i) => ({
                                  x: (binEdges[i] + binEdges[i + 1]) / 2,
                                  y: count || 0,
                                })),
                              },
                            ]}
                            barChartOptions={{
                              chart: {
                                type: 'bar',
                                toolbar: { show: false },
                                zoom: { enabled: false },
                              },
                              plotOptions: {
                                bar: {
                                  distributed: true,
                                  borderRadius: 8,
                                  columnWidth: '12px',
                                },
                              },
                              xaxis: {
                                type: 'numeric',
                                min: binMin,
                                max: binMax,
                                tickAmount: binEdges.length,
                                labels: {
                                  show: false,
                                  style: { colors: '#fff', fontSize: '10px' },
                                  formatter: value => value.toFixed(2),
                                  rotateAlways: true,
                                  rotate: -45, // 라벨을 -45도 회전
                                },
                              },
                              yaxis: {
                                labels: {
                                  style: { colors: '#fff', fontSize: '10px' },
                                },
                              },
                              dataLabels: { enabled: false },
                              legend: { show: false },
                              tooltip: {
                                theme: 'dark',
                                custom: ({
                                  series,
                                  seriesIndex,
                                  dataPointIndex,
                                  w,
                                }) => {
                                  // binEdges 배열은 상위 스코프에서 정의되어 있어야 함.
                                  const startEdge = binEdges[dataPointIndex];
                                  const endEdge = binEdges[dataPointIndex + 1];
                                  const countValue =
                                    series[seriesIndex][dataPointIndex];
                                  return `<div style="padding: 8px; color: #fff; background: #00000080;">
                                            <div><strong>Range:</strong> ${parseFloat(
                                              startEdge
                                            ).toLocaleString()} ~ ${parseFloat(
                                    endEdge
                                  ).toLocaleString()}</div>
                                            <div><strong>Count:</strong> ${parseInt(
                                              countValue
                                            )}</div>
                                          </div>`;
                                },
                                style: { fontSize: '14px' },
                              },
                              colors: colorsArray,
                              annotations: {
                                yaxis: [
                                  {
                                    y: avgValue,
                                    borderColor: 'red',
                                    label: {
                                      position: 'left',
                                      offsetX: 35,
                                      style: {
                                        color: '#fff',
                                        background: '#0c0c0c',
                                        fontSize: '8px',
                                      },
                                    },
                                  },
                                ],
                              },
                            }}
                            width="100%"
                            height="200px"
                          />
                        </Box>
                      );
                    })}
                  </Grid>
                ) : (
                  <Text color="white" textAlign="center">
                    No numeric properties available.
                  </Text>
                )}
                <Flex justify="center" align="center">
                  <IconButton
                    aria-label="Previous Page"
                    icon={<ChevronLeftIcon />}
                    onClick={handlePrevNumericPage}
                    isDisabled={numericPage === 0}
                    mr={2}
                  />
                  <Text color="white" mx={2}>
                    Page {numericPage + 1} of {numericTotalPages}
                  </Text>
                  <IconButton
                    aria-label="Next Page"
                    icon={<ChevronRightIcon />}
                    onClick={handleNextNumericPage}
                    isDisabled={numericPage === numericTotalPages - 1}
                    ml={2}
                  />
                </Flex>
              </TabPanel>

              {/* Categorical Tab */}
              <TabPanel>
                {categoricalCurrentPageProperties.length > 0 ? (
                  <Grid
                    templateColumns="repeat(3, 1fr)"
                    gap={4}
                    w="100%"
                    mb={6}
                  >
                    {categoricalCurrentPageProperties.map(prop => {
                      const hData = histograms[prop];
                      let binEdges = [];
                      let counts = [];
                      if (hData) {
                        try {
                          binEdges = JSON.parse(hData.bin_edges) || [];
                          counts = JSON.parse(hData.counts) || [];
                        } catch (e) {
                          console.error('Histogram parse error:', e);
                        }
                      }
                      // HorizontalBarChart data: 각 bin의 label과 value로 구성
                      let barData = binEdges.map((edge, i) => ({
                        label: edge,
                        value: counts[i] || 0,
                      }));

                      barData = combineCategories(barData, 15);

                      return (
                        <Box
                          key={prop}
                          fontSize="xl"
                          textAlign="center"
                          h="400px"
                          pb={10}
                        >
                          <Text color="gray.300" fontWeight="bold" mb={2}>
                            {prop}
                          </Text>

                          <HorizontalBarChart data={barData} />
                        </Box>
                      );
                    })}
                  </Grid>
                ) : (
                  <Text color="white" fontSize="xl" textAlign="center" p={5}>
                    No categorical properties available.
                  </Text>
                )}
                <Flex justify="center" align="center">
                  <IconButton
                    aria-label="Previous Page"
                    icon={<ChevronLeftIcon />}
                    onClick={handlePrevCategoricalPage}
                    isDisabled={categoricalPage === 0}
                    mr={2}
                  />
                  <Text color="white" mx={2}>
                    Page {categoricalPage + 1} of {categoricalTotalPages}
                  </Text>
                  <IconButton
                    aria-label="Next Page"
                    icon={<ChevronRightIcon />}
                    onClick={handleNextCategoricalPage}
                    isDisabled={categoricalPage === categoricalTotalPages - 1}
                    ml={2}
                  />
                </Flex>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </CardBody>
      </Card>
    );
  };

  const SelectedDatasetsCard = ({ selectedDatasets }) => {
    return (
      <Card w="100%" h="100%">
        <CardHeader>
          <Text color="#fff" fontSize="lg" fontWeight="bold">
            Selected Datasets
          </Text>
        </CardHeader>
        <CardBody
          mt={2}
          overflowY="auto"
          css={{
            '&::-webkit-scrollbar': {
              width: '0px',
            },
          }}
        >
          {selectedDatasets && selectedDatasets.length > 0 ? (
            <Grid
              templateColumns="repeat(auto-fit, minmax(150px, 1fr))"
              gap={2}
            >
              {selectedDatasets.map(ds => (
                <Card key={ds.csvId} p={2} alignItems="center">
                  <Text color="brand.400" fontSize="md" fontWeight="bold">
                    {ds.csv.split('/').pop()}
                  </Text>
                  <Text color="gray.400" fontSize="xs">
                    Rows: {ds.rows.toLocaleString()}
                  </Text>
                  <Text color="gray.400" fontSize="xs">
                    Size: {(ds.size / 1024).toFixed(2)} MB
                  </Text>
                </Card>
              ))}
            </Grid>
          ) : (
            <Text color="white">No dataset selected.</Text>
          )}
        </CardBody>
      </Card>
    );
  };

  const LeftStatsCard = ({
    totalSize, // 현재 플로우 데이터셋 용량 (KB)
    totalCapacity, // 전체 가능한 용량 (MB)
    totalRows, // 전체 rows 수
    numericCount,
    categoricalCount,
    textCount, // 만약 text property가 있다면 (없으면 0)
    unavailableCount,
  }) => {
    const pieData = [
      { label: 'Numeric', value: numericCount },
      { label: 'Categorical', value: categoricalCount },
      { label: 'Text', value: textCount || 0 },
      { label: 'Unavailable', value: unavailableCount },
    ];

    const colorsArray = [
      'rgba(111, 81, 219, 0.77)',
      'rgba(217, 101, 235, 0.6)',
      'rgba(146, 245, 121, 0.6)',
      'rgba(255, 94, 94, 0.5)',
    ];

    totalSize = (parseFloat(totalSize) / 1024).toFixed(2);

    // 계산: dataset 용량 progress, rows progress (백분율)
    const capacityProgress =
      totalCapacity > 0 ? (totalSize / totalCapacity) * 100 : 0;

    return (
      <Grid templateRows="0.5fr 0.5fr 1fr" gap={4} h="100%">
        {/* 카드 1: Dataset Capacity */}
        <Card>
          <CardHeader>
            <Text color="#fff" fontSize="lg" fontWeight="bold">
              Dataset Capacity
            </Text>
          </CardHeader>

          <CardBody
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            mt={2}
          >
            <Box display="flex" mt={3}>
              <Text color="gray.400" fontSize="2xl">
                {totalSize}
              </Text>
              <Text color="#fff" fontSize="2xl" ml={3}>
                / {totalCapacity} MB
              </Text>
            </Box>
            <CircularProgress
              value={capacityProgress}
              size="50px"
              thickness="12px"
              color="brand.100"
            >
              <CircularProgressLabel color="white">
                {Math.round(capacityProgress)}%
              </CircularProgressLabel>
            </CircularProgress>
          </CardBody>
        </Card>

        {/* 카드 2: Total Rows */}
        <Card>
          <CardHeader>
            <Text color="#fff" fontSize="lg" fontWeight="bold">
              Total Rows
            </Text>
          </CardHeader>

          <CardBody
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            mt={2}
          >
            <Box display="flex">
              <Text mt={4} fontSize="2xl" color="white">
                {totalRows.toLocaleString()} Rows
              </Text>
            </Box>
          </CardBody>
        </Card>

        {/* 카드 3: Property Distribution (Pie Chart) */}
        <Card>
          <CardHeader>
            <Text color="#fff" fontSize="lg" fontWeight="bold">
              Property Distribution
            </Text>
          </CardHeader>

          <CardBody
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            mt={0}
          >
            {/* PieChart 컴포넌트를 사용 (가로/세로 크기 조정 필요) */}
            <Box width="160px" height="140px">
              <PieChart data={pieData} colorsArray={colorsArray} />
            </Box>
            {/* Legend 박스 */}
            <Box
              display="flex"
              flexDirection="column"
              justifyContent="space-around"
              height="120px"
              mt={2}
              ml={4} // PieChart와 간격 조정
            >
              {pieData.map((item, i) => (
                <Box
                  key={item.label}
                  display="flex"
                  alignItems="center"
                  bg={colorsArray[i]}
                  borderRadius="md"
                  p={0}
                  mt={3}
                  px={4}
                >
                  <Text color="white" fontSize="xs" fontWeight="semibold">
                    {item.label}
                  </Text>
                </Box>
              ))}
            </Box>
          </CardBody>
        </Card>
        {/* 하단 카드: Selected Datasets (예시) */}
        <SelectedDatasetsCard
          selectedDatasets={selectedDatasets}
        ></SelectedDatasetsCard>
      </Grid>
    );
  };

  return (
    <Flex flexDirection="column" pt={{ base: '120px', md: '75px' }} px={6}>
      {/* 상단 헤더 */}
      <Flex justifyContent="space-between" alignItems="center" mb={6} px={6}>
        <IconButton
          icon={<ArrowBackIcon />}
          onClick={handleGoBack}
          colorScheme="blue"
        />
        <Flex direction="column" align="center">
          <Text fontSize="2xl" fontWeight="bold" color="white">
            Analyze Properties
          </Text>
          <Text fontSize="md" color="gray.400">
            Review and analyze the properties of your datasets to ensure they
            align with your objectives.
          </Text>
        </Flex>
        <IconButton
          icon={<ArrowForwardIcon />}
          colorScheme="blue"
          aria-label="Next Step"
          onClick={handleNextStep}
        />
      </Flex>

      <Grid templateColumns="3fr 1fr" gap={4} h="calc(80vh - 80px)">
        <DistributionCard />
        {/* 상단 영역: Flow 데이터 통계 (LeftStatsCards) */}
        <LeftStatsCard
          totalSize={totalSelectedSize}
          totalCapacity={1024}
          totalRows={totalSelectedRows}
          numericCount={numericCount}
          categoricalCount={categoricalCount}
          textCount={textCount}
          unavailableCount={unavailableCount}
        />
      </Grid>
    </Flex>
  );
}

export default AnalyzePropertiesPage;
