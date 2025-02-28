import React, { useState, useEffect } from 'react';
import {
  Box,
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  Flex,
} from '@chakra-ui/react';
import PropTypes from 'prop-types';
import { useSelector, useDispatch } from 'react-redux';
import { useParams, NavLink, useLocation } from 'react-router-dom';
import { fetchProjects } from 'store/features/projectSlice';
import { fetchFlowsByProject } from 'store/features/flowSlice';

export default function ProjectNavbar(props) {
  const [scrolled, setScrolled] = useState(false);
  const { fixed, secondary, ...rest } = props;

  const location = useLocation();
  const { projectId, flowId } = useParams();
  const dispatch = useDispatch();

  // ✅ Redux 상태 가져오기
  const projects = useSelector(state => state.projects.projects);
  const flows = useSelector(state =>
    Object.values(state.flows.flows).filter(
      flow => flow.projectId?.toString() === projectId?.toString()
    )
  );
  const status = useSelector(state => state.projects.status);

  // ✅ 프로젝트 및 플로우 찾기 (id → projectId, flowId 수정)
  const project = projects.find(
    proj => proj.projectId.toString() === projectId
  );
  const flow = flows.find(flw => flw.flowId.toString() === flowId);

  // ✅ 새로고침 시 데이터 로드
  useEffect(() => {
    if (status === 'idle') {
      dispatch(fetchProjects());
    }
    if (projectId) {
      dispatch(fetchFlowsByProject(projectId));
    }
  }, [dispatch, projectId, status]);

  // ✅ Navbar 스타일
  let navbarPosition = 'absolute';
  let navbarShadow = 'none';
  let navbarBg = 'none';
  let navbarBorder = 'transparent';
  let navbarFilter = 'none';
  let navbarBackdrop = 'none';
  let secondaryMargin = '0px';
  let paddingX = '15px';

  if (fixed && scrolled) {
    navbarPosition = 'fixed';
    navbarShadow = '0px 7px 23px rgba(0, 0, 0, 0.05)';
    navbarBg =
      'linear-gradient(rgba(255, 255, 255, 0) 0% rgba(255, 255, 255, 0.39) @ 100%)';
    navbarBorder = 'rgba(226, 232, 240, 0.3)';
    navbarFilter = 'drop-shadow(0px 7px 23px rgba(0, 0, 0, 0.05))';
    navbarBackdrop = 'blur(42px)';
  }

  const changeNavbar = () => {
    if (window.scrollY > 1) {
      setScrolled(true);
    } else {
      setScrolled(false);
    }
  };

  window.addEventListener('scroll', changeNavbar);

  // ✅ 데이터 로드 중일 경우 Navbar 숨김
  if (status === 'loading') {
    return null;
  }

  return (
    <Flex
      position={navbarPosition}
      boxShadow={navbarShadow}
      bg={navbarBg}
      borderColor={navbarBorder}
      filter={navbarFilter}
      backdropFilter={navbarBackdrop}
      borderWidth="1.5px"
      borderStyle="solid"
      transition="0.25s linear"
      alignItems={{ xl: 'center' }}
      borderRadius="16px"
      display="flex"
      minH="75px"
      justifyContent={{ xl: 'center' }}
      lineHeight="25.6px"
      mx="auto"
      mt={secondaryMargin}
      pb="8px"
      left={document.documentElement.dir === 'rtl' ? '30px' : ''}
      right={document.documentElement.dir === 'rtl' ? '' : '30px'}
      px={{ sm: paddingX, md: '30px' }}
      ps={{ xl: '12px' }}
      pt="8px"
      top="18px"
      w={{ sm: 'calc(100vw - 60px)', xl: 'calc(100vw - 75px - 275px)' }}
    >
      <Flex
        w="100%"
        flexDirection={{ sm: 'column', md: 'row' }}
        alignItems={{ xl: 'center' }}
      >
        <Box mb={{ sm: '8px', md: '0px' }}>
          <Breadcrumb>
            <BreadcrumbItem>
              <BreadcrumbLink
                as={NavLink}
                to="/admin/projects"
                color={
                  location.pathname === '/admin/projects' ? 'white' : '#A0AEC0'
                }
                fontSize="lg"
              >
                Projects
              </BreadcrumbLink>
            </BreadcrumbItem>
            {project && (
              <BreadcrumbItem>
                <BreadcrumbLink
                  as={NavLink}
                  to={`/projects/${project.projectId}`}
                  color={
                    location.pathname === `/projects/${project.projectId}`
                      ? 'white'
                      : '#A0AEC0'
                  }
                  fontSize="lg"
                >
                  {project.name}
                </BreadcrumbLink>
              </BreadcrumbItem>
            )}
            {flow && (
              <BreadcrumbItem>
                <BreadcrumbLink
                  color={
                    location.pathname.includes(
                      `/projects/${projectId}/flows/${flowId}`
                    )
                      ? 'white'
                      : '#A0AEC0'
                  }
                  fontSize="lg"
                >
                  {flow.flow_name}
                </BreadcrumbLink>
              </BreadcrumbItem>
            )}
          </Breadcrumb>
        </Box>
      </Flex>
    </Flex>
  );
}

ProjectNavbar.propTypes = {
  fixed: PropTypes.bool,
  secondary: PropTypes.bool,
};
