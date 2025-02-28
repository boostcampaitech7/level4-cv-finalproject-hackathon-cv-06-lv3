import React, { useEffect, useState } from 'react';
import { Flex, Grid, Box, Text, Spinner, useToast } from '@chakra-ui/react';
import { CheckIcon, TimeIcon } from '@chakra-ui/icons';
import Card from 'components/Card/Card';
import CardHeader from 'components/Card/CardHeader';
import CardBody from 'components/Card/CardBody';
import { useDispatch, useSelector } from 'react-redux';
import { useParams, useHistory } from 'react-router-dom';
import { pollFlowProgress } from 'store/features/flowSlice'; // 경로에 맞게 수정

// 그룹 정의: 각 그룹은 Preprocessing, Surrogate Model, Search Model
const groups = [
  { id: 1, title: 'Preprocessing', startStage: 1, endStage: 2 },
  { id: 2, title: 'Surrogate Model', startStage: 3, endStage: 4 },
  { id: 3, title: 'Search Model', startStage: 5, endStage: 6 },
];

const ModelTrainingProgressPage = () => {
  const { flowId, projectId } = useParams();
  const dispatch = useDispatch();
  const toast = useToast();
  const history = useHistory();

  // redux store에서 flow의 progress를 가져옴
  const currentStage = useSelector(
    state => state.flows.flows[flowId]?.progress
  );
  const [loading, setLoading] = useState(true);

  // 각 그룹별 시작시간 및 완료 시간을 기록하는 state
  const [groupStartTimes, setGroupStartTimes] = useState({});
  const [groupTimes, setGroupTimes] = useState({});
  const [currentStageStart, setCurrentStageStart] = useState(Date.now());
  const [elapsedTime, setElapsedTime] = useState(0);
  const prevStageRef = useState(currentStage)[0]; // 또는 useRef(currentStage)

  // polling: 3초마다 API 호출하여 progress 업데이트 (redux에 저장)
  useEffect(() => {
    if (!flowId) return;
    const intervalId = dispatch(pollFlowProgress(flowId, toast));
    setLoading(false);
    return () => clearInterval(intervalId);
  }, [dispatch, flowId, toast]);

  // 매초 타이머 업데이트 (리렌더링용)
  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime(Date.now() - currentStageStart);
    }, 1000);
    return () => clearInterval(timer);
  }, [currentStageStart]);

  // currentStage 변경 감지: 이전 단계 종료시간 기록 후 타이머 재설정
  useEffect(() => {
    // 이전 currentStage가 존재하고, 새로운 currentStage가 증가했다면 기록
    if (prevStageRef && currentStage > prevStageRef) {
      const finalTime = Date.now() - currentStageStart;
      setGroupTimes(prev => ({ ...prev, [prevStageRef]: finalTime }));
      setCurrentStageStart(Date.now());
      setElapsedTime(0);
    }
    // 업데이트 후 prevStageRef를 갱신 (여기서는 useRef를 사용하면 더 적합함)
    // (이 예제에서는 단순화를 위해 직접 할당하지 않고, currentStage를 참조하는 형태로 사용)
  }, [currentStage, currentStageStart]);

  // currentStage가 6에 도달하면 7초간 toast 후 CheckPerformance 페이지로 이동
  useEffect(() => {
    if (currentStage === 6) {
      toast({
        title: 'Generating Results',
        description:
          'Optimization has been completed successfully. Generating results.',
        status: 'info',
        duration: 7000,
        isClosable: true,
        containerStyle: {
          marginLeft: '280px',
        },
      });
      setTimeout(() => {
        history.push(
          `/projects/${projectId}/flows/${flowId}/check-performance`
        );
      }, 5000);
    }
  }, [currentStage, toast, history, flowId, projectId]);

  // 각 그룹 카드 렌더링 함수
  const renderGroupCard = group => {
    let bgColor = 'gray.700';
    let icon = null;
    let timerText = '';
    if (!currentStage || currentStage < group.startStage) {
      bgColor = 'gray.700';
      timerText = 'Not started';
    } else if (
      currentStage >= group.startStage &&
      currentStage < group.endStage
    ) {
      bgColor = 'yellow.400';
      const startTime = groupStartTimes[group.id];
      if (startTime) {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        timerText = `${elapsed}s elapsed`;
      } else {
        // 그룹 시작 시간이 없으면 현재 타이머 사용
        timerText = `${Math.floor(elapsedTime / 1000)}s elapsed`;
      }
      icon = <TimeIcon boxSize={6} color="white" />;
      // 만약 그룹이 시작되었는데 시작시간이 아직 기록되지 않았다면 기록
      if (!groupStartTimes[group.id]) {
        setGroupStartTimes(prev => ({ ...prev, [group.id]: Date.now() }));
      }
    } else if (currentStage >= group.endStage) {
      bgColor = 'green.500';
      const finalTime = groupTimes[group.id];
      timerText = finalTime
        ? `Completed in ${Math.floor(finalTime / 1000)}s`
        : 'Completed';
      icon = <CheckIcon boxSize={6} color="white" />;
    }

    return (
      <Box
        key={group.id}
        p={4}
        borderRadius="md"
        bg={bgColor}
        textAlign="center"
        minW="200px"
      >
        <Text fontSize="lg" fontWeight="bold" color="white">
          {group.title}
        </Text>
        {timerText && (
          <Text mt={2} fontSize="sm" color="white">
            {timerText}
          </Text>
        )}
        {icon && <Box mt={2}>{icon}</Box>}
      </Box>
    );
  };

  if (loading) {
    return (
      <Flex pt={{ base: '120px', md: '75px' }} justify="center">
        <Spinner color="white" size="xl" />
      </Flex>
    );
  }

  return (
    <Flex
      direction="column"
      p={6}
      minH="80vh"
      pt={{ base: '120px', md: '75px' }}
      align="center"
    >
      <Card w="100%">
        <CardHeader>
          <Text color="white" fontSize="2xl" fontWeight="bold">
            Model Training Progress
          </Text>
        </CardHeader>
        <CardBody>
          <Flex
            justify="center"
            align="center"
            gap={6}
            mt={4}
            w="100%"
            mx="auto"
          >
            {groups.map(group => renderGroupCard(group))}
          </Flex>
        </CardBody>
      </Card>
    </Flex>
  );
};

export default ModelTrainingProgressPage;
