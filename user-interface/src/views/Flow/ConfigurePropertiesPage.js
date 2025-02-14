import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useParams, useHistory } from 'react-router-dom';
import { createPortal } from 'react-dom';
import {
  Flex,
  Grid,
  Box,
  IconButton,
  Divider,
  Text,
  Button,
} from '@chakra-ui/react';
import { ArrowBackIcon, ArrowForwardIcon } from '@chakra-ui/icons';
import {
  DragDropContext,
  Droppable,
  Draggable,
  DroppableProvided,
} from 'react-beautiful-dnd';
import Card from 'components/Card/Card';
import CardBody from 'components/Card/CardBody';
import CardHeader from 'components/Card/CardHeader';

import {
  fetchFlowProperties,
  savePropertyCategories,
  updateCategory,
} from 'store/features/flowSlice';
import { removeCategory } from 'store/features/flowSlice';
import { fetchPropertyTypes } from 'store/features/flowSlice';

/* ------------------------------------------------------
   (A) 개별 property를 Draggable로 표시하는 컴포넌트
------------------------------------------------------ */
function DraggableProperty({
  prop,
  index,
  isDisabled,
  bgColor,
  getOriginalCategory,
}) {
  return (
    <Draggable
      key={prop}
      draggableId={prop}
      index={index}
      isDragDisabled={isDisabled}
    >
      {(provided, snapshot) => {
        const child = (
          <Box
            ref={provided.innerRef}
            {...provided.draggableProps}
            {...provided.dragHandleProps}
            display="inline-flex" // or inline-block
            alignItems="center"
            justifyContent="center"
            px={2}
            py={1}
            bg={bgColor}
            borderRadius="md"
            color="white"
            fontWeight="bold"
            mb={1}
            ml={1}
            mr={1}
            mt={1}
            zIndex={snapshot.isDragging ? 1000 : 'auto'}
            position={snapshot.isDragging ? 'fixed' : 'relative'}
            left={
              snapshot.isDragging
                ? `${provided.draggableProps.style?.left || 0}px`
                : 'auto'
            }
            top={
              snapshot.isDragging
                ? `${provided.draggableProps.style?.top || 0}px`
                : 'auto'
            }
            // 너비를 내용에 맞춤
            width="auto"
            whiteSpace="nowrap"
            // hover 시 크기 커지는 효과 추가
            _hover={{
              transform: 'scale(1.3)',
              transition: 'transform 0.2s ease-in-out',
            }}
          >
            <Text fontSize="xs" fontWeight={'bold'}>
              {prop}
            </Text>
          </Box>
        );

        // 드래그 중일 때는 createPortal로 body에 띄우기
        return snapshot.isDragging ? createPortal(child, document.body) : child;
      }}
    </Draggable>
  );
}

/* ------------------------------------------------------
   (B) 하나의 Droppable 영역을 렌더링하는 컴포넌트
   예: LeftDatasetProperties 안의 numerical/categorical/unavailable
       RightCategorizedProperties 안의 environmental/controllable/output
------------------------------------------------------ */
function DroppableList({
  droppableId,
  title,
  propertiesList,
  getOriginalCategory,
  propertyColors,
  isLeftSide,
}) {
  return (
    <Droppable droppableId={droppableId}>
      {provided => (
        <Box
          ref={provided.innerRef}
          {...provided.droppableProps}
          p={4}
          bg={'transparent'}
          borderRadius="xl"
          boxShadow={'none'}
          border={'none'}
          minH={isLeftSide ? '30px' : '300px'} /* 오른쪽을 조금 더 크게 */
          maxH={isLeftSide ? '60px' : '600px'}
          h="90%"
          minW="100%"
          overflowY="auto"
          css={{
            '&::-webkit-scrollbar': {
              width: '0px',
            },
          }}
        >
          {propertiesList.map((prop, index) => {
            const originalCat = getOriginalCategory(prop);

            const bgColor = propertyColors[originalCat] || 'gray.500';
            const isDisabled = originalCat === 'unavailable' && isLeftSide;

            return (
              <DraggableProperty
                key={prop}
                prop={prop}
                index={index}
                isDisabled={isDisabled}
                bgColor={bgColor}
                getOriginalCategory={getOriginalCategory}
              />
            );
          })}

          {provided.placeholder}
        </Box>
      )}
    </Droppable>
  );
}

/* ------------------------------------------------------
   (C) 왼쪽 섹션: DatasetProperties (numerical, categorical, unavailable)
------------------------------------------------------ */
function LeftDatasetProperties({
  datasetProperties,
  getOriginalCategory,
  propertyColors,
  onMoveAll,
}) {
  // 왼쪽에 표시할 카테고리 목록 (예: ["numerical", "categorical", "unavailable"])
  const categories = Object.keys(datasetProperties);

  // 4단계로 점차 진해지는 그라데이션 배열
  const gradients = [
    'linear-gradient(125deg, rgba(74,81,114,0.5) 0%, rgba(81, 97, 127, 0.9) 20%, rgba(13,23,67,1) 100%)',

    'linear-gradient(125deg, rgba(74,81,114,0.4) 0%, rgba(58, 70, 92, 0.8) 20%, rgba(13,23,67,0.8) 100%)',

    'linear-gradient(125deg, rgba(74,81,114,0.4) 0%, rgba(40, 48, 62, 0.8) 20%, rgba(13,23,67, 0.8) 100%)',

    'linear-gradient(125deg, rgba(74,81,114,0.2) 0%, rgba(56, 67, 88, 0.4) 20%, rgba(13,23,67,0.4) 100%)',
  ];

  return (
    <Card h="100%">
      <CardHeader
        display="flex"
        justifyContent="space-between"
        alignItems="center"
      >
        <Text color="#fff" fontSize="lg" fontWeight="bold">
          Dataset Properties
        </Text>
        <Button size="sm" colorScheme="teal" onClick={onMoveAll}>
          Move ALL to Controllable
        </Button>
      </CardHeader>

      <CardBody h="100%" mt={4}>
        <Grid
          // 카테고리 개수만큼 row 생성
          templateRows="150px 150px 120px 120px"
          gap={3}
          h="100%"
          w="100%"
          alignItems={'start'}
        >
          {categories.map((category, i) => {
            // i: 0,1,2,3,... 배열 범위 넘어가면 마지막(4번째) 톤으로 적용
            const bgGradient = gradients[i] || gradients[gradients.length - 1];

            return (
              <Card key={category} w="100%" h="100%" bg={bgGradient}>
                <CardHeader>
                  <Text color="gray.100" fontWeight="bold" fontSize="md">
                    {category.toUpperCase()}
                  </Text>
                </CardHeader>

                <CardBody
                  h="100%"
                  w="100%"
                  minW="300"
                  overflowY="auto"
                  css={{
                    '&::-webkit-scrollbar': {
                      width: '0px',
                    },
                  }}
                >
                  <DroppableList
                    droppableId={category}
                    title={category}
                    propertiesList={datasetProperties[category] || []}
                    getOriginalCategory={getOriginalCategory}
                    propertyColors={propertyColors}
                    isLeftSide
                  />
                </CardBody>
              </Card>
            );
          })}
        </Grid>
      </CardBody>
    </Card>
  );
}

/* ------------------------------------------------------
   (D) 오른쪽 섹션: CategorizedProperties
       (environmental, controllable, output) 3칸 Grid
------------------------------------------------------ */
function RightCategorizedProperties({
  categorizedProperties,
  getOriginalCategory,
  propertyColors,
}) {
  const categories = Object.keys(categorizedProperties);

  // 3개의 카드에 쓸 그라데이션 (배열 길이를 카테고리 수와 맞추거나, 모자라면 %연산 사용 가능)
  const categoryGradients = [
    // (1) 밝은 톤
    'linear-gradient(125deg, rgba(28,40,51,0.2) 0%, rgba(44,62,80,0.8) 30%, rgba(76,94,112,0.2) 70%)',

    // (2) 중간 톤
    'linear-gradient(125deg, rgba(19,29,39,0.2) 0%, rgba(31,42,57,0.8) 30%, rgba(59,74,90,0.2) 70%)',

    // (3) 어두운 톤
    'linear-gradient(125deg, rgba(11,20,28,0.2) 0%, rgba(24,32,45,0.8) 30%, rgba(42,58,72,0.2) 70%)',
  ];

  const categoryDescriptions = [
    'Used for model training, but cannot be manipulated.',
    'Optimization targets that can be actively adjusted.',
    'Derived from the objective variables identified through the controllable inputs.',
  ];

  return (
    <Card h="100%">
      <CardHeader>
        <Text color="#fff" fontSize="lg" fontWeight="bold">
          Categorized Properties
        </Text>
      </CardHeader>

      <CardBody h="100%" mt={4}>
        {/* 3개의 카테고리를 가로로 배치 (카테고리 수만큼 반복) */}
        <Grid
          templateColumns={`repeat(${categories.length}, 1fr)`}
          gap={4}
          h="100%"
          w="100%"
        >
          {categories.map((category, i) => (
            <Card
              key={category}
              w="100%"
              h="100%"
              // 배열 길이보다 인덱스가 많을 수도 있으니 안전하게 % 연산
              bg={categoryGradients[i % categoryGradients.length]}
            >
              <CardHeader h="20">
                <Flex flexDirection={'column'}>
                  <Text color="gray.100" fontWeight="bold" fontSize="lg">
                    {category.toUpperCase()}
                  </Text>
                  <Text color="gray.500" fontSize="xs" mt={2}>
                    {categoryDescriptions[i % categoryDescriptions.length]}
                  </Text>
                </Flex>
              </CardHeader>

              <CardBody h="100%" mt={3}>
                <DroppableList
                  droppableId={category}
                  title={category}
                  propertiesList={categorizedProperties[category] || []}
                  getOriginalCategory={getOriginalCategory}
                  propertyColors={propertyColors}
                  isLeftSide={false}
                />
              </CardBody>
            </Card>
          ))}
        </Grid>
      </CardBody>
    </Card>
  );
}

/* ------------------------------------------------------
   (E) 메인 페이지 컴포넌트
------------------------------------------------------ */
const ConfigurePropertiesPage = () => {
  const { projectId, flowId } = useParams();
  const dispatch = useDispatch();
  const history = useHistory();

  // (1) 기존 numerical, categorical, unavailable
  const properties = useSelector(
    state =>
      state.flows.properties?.[flowId] || {
        numerical: [],
        categorical: [],
        text: [],
        unavailable: [],
      }
  );

  // (2) 새로운 environmental, controllable, output
  const newCategories = useSelector(
    state => state.flows.newCategories?.[flowId] || {}
  );

  // (3) 로컬 state
  const [datasetProperties, setDatasetProperties] = useState({
    numerical: [],
    categorical: [],
    text: [],
    unavailable: [],
  });
  const [categorizedProperties, setCategorizedProperties] = useState({
    environmental: [],
    controllable: [],
    output: [],
  });

  /* ------------------------------------------------------
     (F) 데이터 Fetch
  ------------------------------------------------------ */
  useEffect(() => {
    dispatch(fetchPropertyTypes(flowId));
  }, [dispatch, flowId]);

  /* ------------------------------------------------------
     (G) 로컬 state 갱신: datasetProperties / categorizedProperties
  ------------------------------------------------------ */
  useEffect(() => {
    // 왼쪽(기존)에서 "새로운 카테고리에 포함되지 않은" 속성만 남기기
    const updatedDatasetProps = {
      numerical: properties.numerical.filter(p => !newCategories[p]),
      categorical: properties.categorical.filter(p => !newCategories[p]),
      text: properties.text.filter(p => !newCategories[p]),
      unavailable: properties.unavailable.filter(p => !newCategories[p]),
    };

    setDatasetProperties(prev =>
      JSON.stringify(prev) !== JSON.stringify(updatedDatasetProps)
        ? updatedDatasetProps
        : prev
    );

    // 오른쪽(새로운 카테고리) 설정
    const updatedCategorizedProps = {
      environmental: Object.keys(newCategories).filter(
        p => newCategories[p] === 'environmental'
      ),
      controllable: Object.keys(newCategories).filter(
        p => newCategories[p] === 'controllable'
      ),
      output: Object.keys(newCategories).filter(
        p => newCategories[p] === 'output'
      ),
    };

    setCategorizedProperties(prev =>
      JSON.stringify(prev) !== JSON.stringify(updatedCategorizedProps)
        ? updatedCategorizedProps
        : prev
    );
  }, [properties, JSON.stringify(newCategories)]);

  /* ------------------------------------------------------
     (H) Drag & Drop 로직
  ------------------------------------------------------ */
  const handleDragEnd = result => {
    const { source, destination } = result;
    if (!destination) return;

    const sourceId = source.droppableId; // numerical / environmental 등
    const destId = destination.droppableId;
    let movedItem = null;

    // (1) 소스가 왼쪽(datasetProperties)
    if (datasetProperties[sourceId]) {
      const newSourceList = [...datasetProperties[sourceId]];
      movedItem = newSourceList.splice(source.index, 1)[0];
      setDatasetProperties(prev => ({
        ...prev,
        [sourceId]: newSourceList,
      }));
    }
    // (2) 소스가 오른쪽(categorizedProperties)
    else if (categorizedProperties[sourceId]) {
      const newSourceList = [...categorizedProperties[sourceId]];
      movedItem = newSourceList.splice(source.index, 1)[0];
      setCategorizedProperties(prev => ({
        ...prev,
        [sourceId]: newSourceList,
      }));
    }
    if (!movedItem) return;

    // (3) 도착지가 오른쪽
    if (destId in categorizedProperties) {
      setCategorizedProperties(prev => ({
        ...prev,
        [destId]: [...prev[destId], movedItem],
      }));
      // Redux에 업데이트
      dispatch(
        updateCategory({ flowId, property: movedItem, category: destId })
      );
    }
    // (4) 도착지가 왼쪽
    else if (destId in datasetProperties) {
      setDatasetProperties(prev => ({
        ...prev,
        [destId]: [...prev[destId], movedItem],
      }));
      dispatch(removeCategory({ flowId, property: movedItem }));
    }
  };

  // ConfigurePropertiesPage 내부
  const handleMoveAllToEnvironmental = () => {
    // unavailable 카테고리를 제외한 나머지 프로퍼티들만 모으기
    const propertiesToMove = ['numerical', 'categorical', 'text'].reduce(
      (acc, key) => acc.concat(datasetProperties[key] || []),
      []
    );

    if (propertiesToMove.length === 0) return; // 옮길게 없으면 바로 리턴

    // 로컬 state 업데이트: 왼쪽 datasetProperties 중 numerical, categorical, text를 비우고,
    // unavailable은 그대로 둔다.
    setDatasetProperties(prev => ({
      ...prev,
      numerical: [],
      categorical: [],
      text: [],
    }));

    setCategorizedProperties(prev => ({
      ...prev,
      controllable: [...prev.environmental, ...propertiesToMove],
    }));

    // 각 property에 대해 redux 액션 dispatch
    propertiesToMove.forEach(prop => {
      dispatch(
        updateCategory({ flowId, property: prop, category: 'controllable' })
      );
    });
  };

  /* ------------------------------------------------------
     (I) Next Step 처리
  ------------------------------------------------------ */
  const handleNextStep = async () => {
    const updates = Object.keys(newCategories).map(property => ({
      flow_id: parseInt(flowId),
      column_name: property,
      property_type: newCategories[property],
    }));

    // 실제 API 요청 (각 property에 대해)
    for (const update of updates) {
      await dispatch(savePropertyCategories({ flowId, update }));
    }

    history.push(`/projects/${projectId}/flows/${flowId}/set-goals`);
  };

  /* ------------------------------------------------------
     (J) 기타 부가 로직
  ------------------------------------------------------ */
  const getOriginalCategory = prop => {
    if (properties.numerical.includes(prop)) return 'numerical';
    if (properties.categorical.includes(prop)) return 'categorical';
    if (properties.text.includes(prop)) return 'text';
    if (properties.unavailable.includes(prop)) return 'unavailable';
    return 'unavilable';
  };

  const propertyColors = {
    numerical: 'rgba(111, 81, 219, 0.77)',
    categorical: 'rgba(217, 101, 235, 0.77)',
    text: 'rgba(0, 202, 178, 0.77)',
    unavailable: 'rgba(255, 94, 94, 0.77)',
    unknown: 'gray.100',
  };

  /* ------------------------------------------------------
     (K) 레이아웃 렌더
  ------------------------------------------------------ */
  const renderHeader = () => (
    <Flex justifyContent="space-between" alignItems="center" mb={6} px={6}>
      {/* 왼쪽: 뒤로가기 */}
      <IconButton
        icon={<ArrowBackIcon />}
        onClick={() => history.goBack()}
        colorScheme="blue"
      />

      {/* 중앙: 제목/설명 */}
      <Flex direction="column" align="center">
        <Text fontSize="2xl" fontWeight="bold" color="white">
          Configure Properties
        </Text>
        <Text fontSize="md" color="gray.400">
          Drag and drop properties into appropriate categories.
        </Text>
      </Flex>

      {/* 오른쪽: Next Step */}
      <IconButton
        icon={<ArrowForwardIcon />}
        colorScheme="blue"
        aria-label="Next Step"
        onClick={handleNextStep}
      />
    </Flex>
  );

  return (
    <Flex flexDirection="column" pt={{ base: '120px', md: '75px' }} px={6}>
      {renderHeader()}

      <DragDropContext onDragEnd={handleDragEnd}>
        <Grid templateColumns="1fr 1.5fr" h="calc(80vh - 100px)" gap={6} px={6}>
          {/* 왼쪽: 기존 properties */}
          <LeftDatasetProperties
            datasetProperties={datasetProperties}
            getOriginalCategory={getOriginalCategory}
            propertyColors={propertyColors}
            onMoveAll={handleMoveAllToEnvironmental}
          />

          {/* 오른쪽: 새 카테고리 (environmental, controllable, output) */}
          <RightCategorizedProperties
            categorizedProperties={categorizedProperties}
            getOriginalCategory={getOriginalCategory}
            propertyColors={propertyColors}
          />
        </Grid>
      </DragDropContext>
    </Flex>
  );
};

export default ConfigurePropertiesPage;
