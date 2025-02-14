import React, { useState, useEffect } from 'react';
import {
  Box,
  Divider,
  Flex,
  Grid,
  IconButton,
  Text,
  Tooltip,
  useDisclosure,
} from '@chakra-ui/react';
import Card from 'components/Card/Card.js';
import CardBody from 'components/Card/CardBody.js';
import CardHeader from 'components/Card/CardHeader';
import AddFlowDialog from 'components/Dialog/AddFlowDialog';
import EditFlowDialog from 'components/Dialog/EditFlowDialog';
import FlowRow from 'components/Tables/FlowRow';
import { AddIcon } from '@chakra-ui/icons';
import { useParams, useHistory } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import {
  fetchFlowsByProject,
  addFlowAsync,
  deleteFlowAsync,
  editFlowAsync,
} from 'store/features/flowSlice'; // ✅ 경로 수정
import FlowProgressIndicator from 'components/Indicator/FlowProgressIndicator';
import FlowProgressBar from 'components/Indicator/FlowProgressBar';

export default function Project() {
  const { id: projectId } = useParams();
  const dispatch = useDispatch();
  const history = useHistory();

  // ✅ Redux에서 flows를 배열로 변환하여 가져오기 (projectId 기준 필터링)
  const flows = useSelector(state =>
    Object.values(state.flows.flows).filter(
      flow => flow.projectId.toString() === projectId
    )
  );

  const { isOpen, onOpen, onClose } = useDisclosure();
  const {
    isOpen: isEditOpen,
    onOpen: onEditOpen,
    onClose: onEditClose,
  } = useDisclosure();

  const [currentFlow, setCurrentFlow] = useState(null);

  // ✅ 프로젝트의 Flows 가져오기 (초기 렌더링 시)
  useEffect(() => {
    dispatch(fetchFlowsByProject(projectId));
  }, [dispatch, projectId]);

  // ✅ Flow 클릭 시 데이터셋 선택 페이지로 이동
  const handleFlowClick = flowId => {
    history.push(`/projects/${projectId}/flows/${flowId}/select-datasets`);
  };

  // ✅ Flow 추가
  const handleFlowAdd = async newFlow => {
    const response = await dispatch(
      addFlowAsync({ projectId, flowName: newFlow.name })
    ).unwrap();

    if (response && response.flowId) {
      console.log('✅ Flow 추가 완료:', response);
    }

    onClose();
  };

  // ✅ Flow 삭제
  const handleFlowDelete = async flowId => {
    await dispatch(deleteFlowAsync({ flowId })).unwrap();
  };

  // ✅ Flow 수정 모달 열기 (flows를 배열로 변환 후 find() 사용)
  const handleFlowEdit = flowId => {
    const flow = Object.values(flows).find(flw => flw.flowId === flowId);
    setCurrentFlow(flow);
    onEditOpen();
  };

  // ✅ Flow 업데이트
  const handleFlowUpdate = async (flowId, updatedFlowName) => {
    if (!flowId) {
      console.error('❌ Flow ID is missing');
      return;
    }

    console.log('🛠 Updating flow in Redux:', { flowId, updatedFlowName });

    await dispatch(
      editFlowAsync({
        flowId,
        flowName: updatedFlowName,
      })
    ).unwrap();

    setCurrentFlow(null); // ✅ 상태 정리
    onEditClose();
  };

  return (
    <Flex flexDirection="column" pt={{ base: '120px', md: '75px' }}>
      <Card w="100%" h="calc(80vh - 50px)">
        <CardHeader
          mb="16px"
          display="flex"
          justifyContent="space-between"
          alignItems="center"
        >
          <Text fontSize="2xl" fontWeight="bold" color="#fff">
            Project Flows
          </Text>
          <Tooltip label="Add New Flow" aria-label="Add New Flow Tooltip">
            <IconButton
              size="md"
              variant="solid"
              bg="teal.500"
              aria-label="Add New Flow"
              icon={<AddIcon color="#fff" />}
              onClick={onOpen}
              _hover={{ bg: 'teal.500' }}
              _active={{ bg: 'teal.600' }}
            />
          </Tooltip>
        </CardHeader>

        <CardBody
          overflowY="auto"
          css={{
            '&::-webkit-scrollbar': {
              width: '0px',
            },
          }}
          mt={2}
        >
          <Box w="100%">
            <Grid templateColumns="repeat(3, 1fr)" gap={6}>
              <Card
                transition="background-color 0.3s ease"
                _hover={{ bg: 'rgba(35, 40, 65, 0.49)' }}
                cursor="pointer"
                onClick={onOpen}
                display="flex"
                alignItems="center"
                justifyContent="center"
                boxShadow="md"
                w="100%"
              >
                <Text fontSize="md" fontWeight="bold" color="brand.100">
                  🌊 ADD New Flow
                </Text>
              </Card>
              {flows.map(flow => (
                <Box
                  key={flow.flowId}
                  p="16px"
                  bg="transparent"
                  transition="background-color 0.3s ease, transform 0.2s ease"
                  cursor="pointer"
                  boxShadow="md"
                  w="100%"
                  _hover={{
                    //bg: "rgba(6, 12, 41, 0.8)",
                    transform: 'scale(1.01)',
                  }}
                  onClick={() => handleFlowClick(flow.flowId)}
                >
                  <FlowRow
                    name={flow.flow_name}
                    onEdit={() => handleFlowEdit(flow.flowId)}
                    onDelete={() => handleFlowDelete(flow.flowId)}
                    // 필요시 추가 props 전달
                  />
                  {/* FlowProgressBar를 카드 하단에 배치 */}
                  <FlowProgressBar progress={flow.progress || 0} />
                </Box>
              ))}
            </Grid>
          </Box>
        </CardBody>
      </Card>
      <AddFlowDialog
        isOpen={isOpen}
        onClose={onClose}
        onAdd={newFlow => handleFlowAdd(newFlow)}
      />
      {currentFlow && (
        <EditFlowDialog
          isOpen={isEditOpen}
          onClose={onEditClose}
          flow={currentFlow}
          onUpdate={handleFlowUpdate}
        />
      )}
    </Flex>
  );
}
