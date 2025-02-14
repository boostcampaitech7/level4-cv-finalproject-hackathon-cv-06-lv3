import React, { useEffect, useState, useMemo, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useParams, useHistory } from 'react-router-dom';
import { createPortal } from 'react-dom';
import {
  Box,
  Flex,
  Text,
  Button,
  Divider,
  IconButton,
  Badge,
  AspectRatio,
  Grid,
  CircularProgress,
} from '@chakra-ui/react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import { ArrowForwardIcon, ArrowBackIcon } from '@chakra-ui/icons';
import Card from 'components/Card/Card';
import CardBody from 'components/Card/CardBody';
import CardHeader from 'components/Card/CardHeader';
import {
  fetchFlowProperties,
  updatePriorities,
  initializePriorities,
  fetchOptimizationData,
} from 'store/features/flowSlice';
import { postOptimizationOrder } from 'store/features/flowSlice';
import { createModelThunk } from 'store/features/flowSlice';

const SetPrioritiesPage = () => {
  const { projectId, flowId } = useParams();
  const dispatch = useDispatch();
  const history = useHistory();

  // Redux: newCategories (property -> category)
  const properties = useSelector(
    state => state.flows.newCategories[flowId] || {}
  );

  // Fetch flow properties on mount
  useEffect(() => {
    dispatch(fetchFlowProperties(flowId));
  }, [dispatch, flowId]);

  // controllable, output 속성에 대해서만 optimizationData fetch
  useEffect(() => {
    Object.keys(properties).forEach(property => {
      const type = properties[property];
      if (type === 'controllable' || type === 'output') {
        dispatch(fetchOptimizationData({ flowId, property, type }));
      }
    });
  }, [dispatch, flowId, properties]);

  const optimizationData = useSelector(
    state => state.flows.optimizationData[flowId] || {}
  );
  const storedPriorities = useSelector(
    state => state.flows.priorities[flowId] || []
  );

  // controllable과 output property 분리
  const controllableProperties = useMemo(() => {
    return Object.entries(properties)
      .filter(([_, cat]) => cat === 'controllable')
      .map(([prop]) => prop);
  }, [properties]);

  const outputProperties = useMemo(() => {
    return Object.entries(properties)
      .filter(([_, cat]) => cat === 'output')
      .map(([prop]) => prop);
  }, [properties]);

  const [priorities, setPriorities] = useState([]);
  const [isPreparing, setIsPreparing] = useState(false); // 모델 생성 준비 상태

  useEffect(() => {
    if (storedPriorities.length > 0) {
      setPriorities(storedPriorities);
    }
  }, [storedPriorities]);

  const initializedRef = useRef(false);
  useEffect(() => {
    if (!initializedRef.current && Object.keys(optimizationData).length > 0) {
      // optimizationData에 있는 각 property의 order 값 기준으로 정렬 (order 값이 없으면 기본값 0으로 처리)
      const sortedPriorities = Object.keys(optimizationData).sort((a, b) => {
        const orderA = optimizationData[a].order || 0;
        const orderB = optimizationData[b].order || 0;
        return orderA - orderB;
      });
      if (sortedPriorities.length > 0) {
        setPriorities(sortedPriorities);
      } else {
        const defaultPriorities = [
          ...controllableProperties,
          ...outputProperties,
        ];
        setPriorities(defaultPriorities);
      }
      initializedRef.current = false;
    }
  }, [optimizationData, controllableProperties, outputProperties]);

  useEffect(() => {
    dispatch(initializePriorities({ flowId }));
  }, [dispatch, flowId]);

  const onDragEnd = result => {
    if (!result.destination) return;
    const newPriorities = Array.from(priorities);
    const [movedItem] = newPriorities.splice(result.source.index, 1);
    newPriorities.splice(result.destination.index, 0, movedItem);
    setPriorities(newPriorities);
    dispatch(updatePriorities({ flowId, priorities: newPriorities }));
  };

  // 각 goal에 따른 색상 결정 함수 - 어두운 배경에 맞춰 색상을 조금 더 진하게 조정
  const getGoalColor = goal => {
    switch (goal) {
      case 'No Optimization':
        return 'gray.200';
      case 'Maximize':
        return 'green.200';
      case 'Minimize':
        return 'red.200';
      case 'Fit to Range':
        return 'orange.200';
      case 'Fit to Property':
        return 'purple.200';
      default:
        return 'gray.200';
    }
  };

  const handleNextStep = async () => {
    try {
      setIsPreparing(true);
      await dispatch(postOptimizationOrder({ flowId, priorities })).unwrap();

      // 2. 모든 optimizationData POST 요청이 끝난 후, 모델 생성 API 요청 (createModelThunk)
      dispatch(createModelThunk(flowId)).unwrap();
      // 정보 메시지를 1초 정도 보여준 후 다음 페이지로 이동
      setTimeout(() => {
        history.push(
          `/projects/${projectId}/flows/${flowId}/model-training-progress`
        );
      }, 1000);
    } catch (error) {
      console.error('Failed to post optimization orders:', error);
      setIsPreparing(false);
    }
  };

  const renderDraggableCard = (prop, index) => {
    const data = optimizationData[prop] || {};
    const textColor = getGoalColor(data.goal);
    return (
      <Draggable key={prop} draggableId={prop} index={index}>
        {(provided, snapshot) => {
          const child = (
            <Card
              ref={provided.innerRef}
              {...provided.draggableProps}
              {...provided.dragHandleProps}
              bg={
                snapshot.isDragging
                  ? 'gray.900'
                  : 'linear-gradient(126.97deg, #060C29 35.26%, rgba(4, 12, 48, 0.5) 70.2%)'
              }
              p={2}
              maxH="300"
            >
              <AspectRatio ratio={1} w="100%">
                <Flex
                  direction="column"
                  h="100%"
                  justify="center"
                  align="center"
                  maxH="200"
                >
                  <CardHeader p={1}>
                    <Flex direction="column">
                      <Flex align="center">
                        <Badge mr={2} colorScheme="teal">
                          {index + 1}
                        </Badge>
                        <Text
                          fontSize="md"
                          fontWeight="bold"
                          color="white"
                          maxW="150px"
                          isTruncated
                        >
                          {prop}
                        </Text>
                      </Flex>
                      <Text fontSize="xs" color="gray.400" mt={0} ml={0}>
                        {data.type}
                      </Text>
                    </Flex>
                  </CardHeader>

                  <CardBody p={1} mt={4}>
                    <Flex direction="column">
                      {data.goal === 'Fit to Property' ? (
                        <Text fontSize="xs" color="white">
                          {data.minimum_value !== ' '
                            ? data.minimum_value
                            : 'NaN'}
                        </Text>
                      ) : (
                        <Text fontSize="xs" color="white">
                          {data.minimum_value !== ' '
                            ? data.minimum_value
                            : 'NaN'}{' '}
                          ~{' '}
                          {data.maximum_value !== ' '
                            ? data.maximum_value
                            : 'NaN'}
                        </Text>
                      )}
                      <Text fontSize="xs" color={textColor}>
                        {data.goal || '-'}
                      </Text>
                    </Flex>
                  </CardBody>
                </Flex>
              </AspectRatio>
            </Card>
          );
          return snapshot.isDragging
            ? createPortal(child, document.body)
            : child;
        }}
      </Draggable>
    );
  };

  return (
    <Flex
      flexDirection="column"
      pt={{ base: '120px', md: '75px' }}
      px={6}
      color="white"
    >
      {/* 헤더 영역 */}
      <Flex justifyContent="space-between" alignItems="center" mb={6} px={6}>
        <IconButton
          icon={<ArrowBackIcon />}
          onClick={() => history.goBack()}
          colorScheme="blue"
        />
        <Box textAlign="center">
          <Text fontSize="2xl" fontWeight="bold" color="white">
            Set Priorities
          </Text>
          <Text fontSize="md" color="gray.400">
            Drag and drop property cards to configure your optimization goals
            and priorities for model preparation.
          </Text>
        </Box>
        <IconButton
          icon={<ArrowForwardIcon />}
          onClick={handleNextStep}
          colorScheme="blue"
        />
      </Flex>

      {/* 드래그 가능한 카드들을 감싸는 큰 영역 (수평 스크롤) */}
      <Box mb={6} py={50}>
        <DragDropContext onDragEnd={onDragEnd}>
          <Droppable droppableId="priorities" direction="horizontal">
            {provided => (
              <Flex
                ref={provided.innerRef}
                {...provided.droppableProps}
                gap={4}
                justify="center"
              >
                {priorities.map((prop, index) => (
                  <Box key={prop} w="150px">
                    {renderDraggableCard(prop, index)}
                  </Box>
                ))}
                {provided.placeholder}
              </Flex>
            )}
          </Droppable>
        </DragDropContext>
      </Box>

      {isPreparing && (
        <Flex justifyContent="center" alignItems="center" mb={6}>
          <Card
            mt={4}
            bg="linear-gradient(90deg, #171923, #2D3748, #4A5568, #718096)"
            textAlign="center"
            w="50%"
          >
            <Flex align="center" justify="center" gap={3}>
              <Box>
                <CircularProgress isIndeterminate size="30px" color="red.200" />
                <Text fontSize="sm" mt={3} color="red.200">
                  Please wait a moment.
                </Text>
                <Text fontSize="sm" color="red.200">
                  Preparing the model based on your configured goals and
                  priorities.
                </Text>
              </Box>
            </Flex>
          </Card>
        </Flex>
      )}
    </Flex>
  );
};

export default SetPrioritiesPage;
