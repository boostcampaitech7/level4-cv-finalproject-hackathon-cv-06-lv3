import React, { useEffect } from 'react';
import { ChakraProvider } from '@chakra-ui/react';
import Footer from 'components/Footer/Footer.js';
import theme from 'theme/themeAdmin.js';
import MainPanel from 'components/Layout/MainPanel';
import PanelContainer from 'components/Layout/PanelContainer';
import PanelContent from 'components/Layout/PanelContent';
import ProjectNavbar from 'components/Navbars/ProjectNavbar';
import SidebarTimeline from 'components/Sidebar/SidebarTimeline';

// Page Components
import SelectDatasetsPage from 'views/Flow/SelectDatasetsPage';

// React Router imports
import { Switch, Route, useParams } from 'react-router-dom';

// Redux imports
import { useDispatch, useSelector } from 'react-redux';
import { initializeFlow, setCurrentStep } from 'store/features/flowSlice';
import AnalyzePropertiesPage from 'views/Flow/AnalyzePropertiesPage';
import ConfigurePropertiesPage from 'views/Flow/ConfigurePropertiesPage';
import SetGoalsPage from 'views/Flow/SetGoalsPage';
import SetPrioritiesPage from 'views/Flow/SetPrioritiesPage';
import CheckPerformancePage from 'views/Flow/CheckPerformancePage';
import OptimizationResultsPage from 'views/Flow/OptimizationResultsPage';
import ModelTrainingProgressPage from 'views/Flow/ModelTrainingProgressPage';

export default function FlowLayout() {
  const { id: projectId, flowId } = useParams();
  const dispatch = useDispatch();

  const flowState = useSelector(
    state => state.flows.flows[flowId] || { currentStep: 0 }
  );

  useEffect(() => {
    dispatch(initializeFlow({ flowId }));
  }, [dispatch, flowId]);

  const handleStepChange = step => {
    dispatch(setCurrentStep({ flowId, step }));
  };

  return (
    <ChakraProvider theme={theme} resetCss={false}>
      <SidebarTimeline
        currentStep={flowState.currentStep}
        onStepChange={handleStepChange}
      />
      <MainPanel
        w={{
          base: '100%',
          xl: 'calc(100% - 275px)',
        }}
      >
        <ProjectNavbar />
        <PanelContent>
          <PanelContainer>
            <Switch>
              <Route
                path="/projects/:projectId/flows/:flowId/select-datasets"
                component={SelectDatasetsPage}
              />
              <Route
                path="/projects/:projectId/flows/:flowId/analyze-properties"
                component={AnalyzePropertiesPage}
              />
              <Route
                path="/projects/:projectId/flows/:flowId/configure-properties"
                component={ConfigurePropertiesPage}
              />
              <Route
                path="/projects/:projectId/flows/:flowId/set-goals"
                component={SetGoalsPage}
              />
              <Route
                path="/projects/:projectId/flows/:flowId/set-priorities"
                component={SetPrioritiesPage}
              />
              <Route
                path="/projects/:projectId/flows/:flowId/model-training-progress"
                component={ModelTrainingProgressPage}
              />
              <Route
                path="/projects/:projectId/flows/:flowId/check-performance"
                component={CheckPerformancePage}
              />
              <Route
                path="/projects/:projectId/flows/:flowId/optimization-results"
                component={OptimizationResultsPage}
              />
            </Switch>
          </PanelContainer>
        </PanelContent>
        <Footer />
      </MainPanel>
    </ChakraProvider>
  );
}
