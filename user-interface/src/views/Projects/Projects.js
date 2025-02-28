import React, { useState, useEffect } from 'react';
import {
  Box,
  Divider,
  Flex,
  IconButton,
  Text,
  Tooltip,
  VStack,
  useDisclosure,
} from '@chakra-ui/react';
import Card from 'components/Card/Card.js';
import CardBody from 'components/Card/CardBody.js';
import CardHeader from 'components/Card/CardHeader.js';
import ProjectRow from 'components/Tables/ProjectRow';
import AddProjectDialog from 'components/Dialog/AddProjectDialog';
import EditProjectDialog from 'components/Dialog/EditProjectDialog';
import { AddIcon } from '@chakra-ui/icons';
import { useHistory } from 'react-router-dom';

import { useSelector, useDispatch } from 'react-redux';
import {
  fetchProjects,
  addProjectAsync,
  editProjectAsync,
  deleteProjectAsync,
} from 'store/features/projectSlice';

export default function Projects() {
  const dispatch = useDispatch();
  const history = useHistory();

  // Redux 상태
  const projects = useSelector(state => state.projects.projects || []);
  const status = useSelector(state => state.projects.status);
  const error = useSelector(state => state.projects.error);

  // 모달 상태 관리
  const { isOpen, onOpen, onClose } = useDisclosure();
  const {
    isOpen: isEditOpen,
    onOpen: onEditOpen,
    onClose: onEditClose,
  } = useDisclosure();
  const [currentProject, setCurrentProject] = useState(null);

  // 프로젝트 목록 불러오기 (최초 실행)
  useEffect(() => {
    if (status === 'idle') {
      dispatch(fetchProjects())
        .unwrap()
        .then(data => console.log('✅ 프로젝트 데이터 가져옴:', data))
        .catch(error => console.error('❌ 프로젝트 가져오기 실패:', error));
    }
  }, [status, dispatch]);

  // 프로젝트 추가 (API 요청 후 상태 업데이트)
  const handleProjectAdd = async newProject => {
    try {
      await dispatch(addProjectAsync(newProject)).unwrap();
      onClose();
    } catch (err) {
      console.error('❌ 프로젝트 추가 실패:', err);
    }
  };

  // 프로젝트 삭제 (API 요청 후 상태 업데이트)
  const handleProjectDelete = async projectId => {
    try {
      await dispatch(deleteProjectAsync({ projectId })).unwrap();
    } catch (err) {
      console.error('❌ 프로젝트 삭제 실패:', err);
    }
  };

  // 프로젝트 수정 모달 열기
  const handleProjectEdit = projectId => {
    const project = projects.find(proj => proj.projectId === projectId);
    setCurrentProject(project);
    onEditOpen();
  };

  // 프로젝트 수정 (API 요청 후 상태 업데이트)
  const handleProjectUpdate = async updatedProject => {
    try {
      await dispatch(editProjectAsync(updatedProject)).unwrap();
      onEditClose();
    } catch (err) {
      console.error('❌ 프로젝트 수정 실패:', err);
    }
  };

  // 프로젝트 클릭 시 상세 페이지 이동
  const handleProjectClick = projectId => {
    history.push(`/projects/${projectId}`);
  };

  // 날짜 포맷 함수 (영어 형식, 예: "Feb 3, 2025, 2:59 PM")
  const formatDate = dateString => {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <Flex flexDirection="column" pt={{ base: '120px', md: '75px' }} px={4}>
      <Card w="100%">
        <CardHeader
          mb="16px"
          display="flex"
          justifyContent="space-between"
          alignItems="center"
        >
          <Text fontSize="2xl" fontWeight="bold" color="#fff">
            Projects
          </Text>
          <Tooltip label="Add New Project" aria-label="Add New Project Tooltip">
            <IconButton
              size="md"
              variant="solid"
              bg="teal.400"
              aria-label="Add New Project"
              icon={<AddIcon color="#fff" />}
              onClick={onOpen}
              _hover={{ bg: 'teal.500' }}
              _active={{ bg: 'teal.600' }}
            />
          </Tooltip>
        </CardHeader>

        <CardBody mt={3}>
          {error ? (
            <Text color="red.500" textAlign="center">
              Error: {error}
            </Text>
          ) : (
            <Box w="100%">
              <VStack spacing={4} w="100%">
                {projects.length > 0 ? (
                  projects.map(project => (
                    <Card
                      key={project.projectId}
                      bg="linear-gradient(127.09deg, rgba(24, 29, 60, 0.94) 19.41%, rgba(10, 14, 35, 0.49) 76.65%)"
                      transition="background-color 0.3s ease, transform 0.2s ease"
                      cursor="pointer"
                      w="100%"
                      p={4}
                      _hover={{
                        bg: 'rgba(6, 12, 41, 0.8)',
                        transform: 'scale(1.01)',
                      }}
                      onClick={() => handleProjectClick(project.projectId)}
                    >
                      <ProjectRow
                        name={project?.name || 'No Name'}
                        description={project?.description || 'No Description'}
                        isLocked={project?.isLocked || false}
                        onEdit={() => handleProjectEdit(project.projectId)}
                        onDelete={() => handleProjectDelete(project.projectId)}
                      />
                      {/* 생성일 정보를 description 아래에 표시 */}
                      <Text fontSize="xs" color="gray.300" mt={2}>
                        Created at: {formatDate(project.created_at)}
                      </Text>
                    </Card>
                  ))
                ) : (
                  <Text color="gray.400" textAlign="center">
                    No projects available.
                  </Text>
                )}
                {/* 프로젝트 추가 버튼 */}
                <Card
                  boxShadow="md"
                  w="100%"
                  cursor="pointer"
                  onClick={onOpen}
                  transition="background-color 0.3s ease, transform 0.2s ease"
                  _hover={{
                    bg: 'rgba(6, 12, 41, 0.8)',
                    transform: 'scale(1.01)',
                  }}
                >
                  <Flex justify="center" align="center" w="100%" h="100%">
                    <Text fontSize="md" fontWeight="bold" color="brand.100">
                      📁 ADD New Project
                    </Text>
                  </Flex>
                </Card>
              </VStack>
            </Box>
          )}
        </CardBody>
      </Card>
      {/* 프로젝트 추가 모달 */}
      <AddProjectDialog
        isOpen={isOpen}
        onClose={onClose}
        onAdd={handleProjectAdd}
      />
      {/* 프로젝트 수정 모달 */}
      {currentProject && (
        <EditProjectDialog
          isOpen={isEditOpen}
          onClose={onEditClose}
          project={currentProject}
          onUpdate={handleProjectUpdate}
        />
      )}
    </Flex>
  );
}
