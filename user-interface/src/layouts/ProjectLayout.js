import React, { useState } from 'react';
import { ChakraProvider } from '@chakra-ui/react';
import Footer from 'components/Footer/Footer.js';
import { Switch, Route, Redirect, useParams } from 'react-router-dom';
import Project from 'views/Project/Project';
import theme from 'theme/themeAdmin.js';
import MainPanel from 'components/Layout/MainPanel';
import PanelContainer from 'components/Layout/PanelContainer';
import PanelContent from 'components/Layout/PanelContent';
import Sidebar from 'components/Sidebar/Sidebar';
import ProjectNavbar from 'components/Navbars/ProjectNavbar'; // 새로운 Navbar

import { useSelector } from 'react-redux';
import { CarIcon } from 'components/Icons/Icons';

export default function ProjectLayout() {
  // Redux에서 프로젝트 목록 가져오기
  const projects = useSelector(state => state.projects.projects);
  const [sidebarVariant, setSidebarVariant] = useState('transparent');

  // URL에서 현재 프로젝트 ID 가져오기
  const { id } = useParams();
  const currentProject = projects.find(project => project.id === id);

  // Sidebar에 표시할 프로젝트 목록
  const projectRoutes = projects.map(project => ({
    path: `/${project.projectId}`,
    name: project.name,
    layout: '/projects',
    icon: <CarIcon color="inherit"></CarIcon>,
  }));

  return (
    <ChakraProvider theme={theme} resetCss={false}>
      <Sidebar
        routes={projectRoutes}
        logoText="PROJECTS"
        variant={sidebarVariant}
      />
      <MainPanel
        w={{
          base: '100%',
          xl: 'calc(100% - 275px)',
        }}
      >
        {/* ProjectNavbar에 현재 프로젝트 이름 전달 */}
        <ProjectNavbar
          projectName={currentProject ? currentProject.name : 'Default Project'}
        />
        <PanelContent>
          <PanelContainer>
            <Switch>
              {/* 동적 라우트 */}
              <Route path="/projects/:id" component={Project} />
              {/* 잘못된 경로 기본 리다이렉트 */}
              <Redirect from="/projects" to="/admin/projects" />
            </Switch>
          </PanelContainer>
        </PanelContent>
        <Footer />
      </MainPanel>
    </ChakraProvider>
  );
}
