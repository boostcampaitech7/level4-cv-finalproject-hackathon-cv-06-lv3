import React from 'react';
import { Box, Text, Flex, Stack, Divider } from '@chakra-ui/react';
import { Link, useLocation, useParams } from 'react-router-dom';
import { SimmmpleLogoWhite } from 'components/Icons/Icons';
import { SimmmpleEmojiIcon } from 'components/Icons/Icons';

const SidebarTimeline = () => {
  const { projectId, flowId } = useParams(); // 현재 URL에서 projectId와 flowId 가져오기
  const location = useLocation(); // 현재 URL 경로 가져오기

  const steps = [
    {
      title: 'Datasets',
      subSteps: [
        {
          name: 'Select Datasets',
          path: `/projects/${projectId}/flows/${flowId}/select-datasets`,
        },
      ],
    },
    {
      title: 'Properties',
      subSteps: [
        {
          name: 'Analyze Properties',
          path: `/projects/${projectId}/flows/${flowId}/analyze-properties`,
        },
        {
          name: 'Configure Properties',
          path: `/projects/${projectId}/flows/${flowId}/configure-properties`,
        },
      ],
    },
    {
      title: 'Models',
      subSteps: [
        {
          name: 'Set Goals',
          path: `/projects/${projectId}/flows/${flowId}/set-goals`,
        },
        {
          name: 'Set Priorities',
          path: `/projects/${projectId}/flows/${flowId}/set-priorities`,
        },
      ],
    },
    {
      title: 'Results',
      subSteps: [
        {
          name: 'Check Performance',
          path: `/projects/${projectId}/flows/${flowId}/check-performance`,
        },
        {
          name: 'Optimization Results',
          path: `/projects/${projectId}/flows/${flowId}/optimization-results`,
        },
      ],
    },
  ];

  // 현재 URL에 기반하여 활성화된 Step 및 SubStep 계산
  const getCurrentStepAndSubStep = () => {
    let currentStep = -1;
    let currentSubStep = -1;

    steps.forEach((step, stepIndex) => {
      step.subSteps.forEach((subStep, subIndex) => {
        if (location.pathname === subStep.path) {
          currentStep = stepIndex;
          currentSubStep = subIndex;
        }
      });
    });

    return { currentStep, currentSubStep };
  };

  const { currentStep, currentSubStep } = getCurrentStepAndSubStep();

  return (
    <Box
      bg="linear-gradient(111.84deg, rgba(6, 11, 38, 0.94) 59.3%, rgba(26, 31, 55, 0) 100%)"
      display={{ sm: 'none', xl: 'block' }}
      position="fixed"
      h="calc(100vh - 32px)"
      w="260px"
      m="16px"
      p={4}
      pl="20px"
      pt="25px"
      borderRadius="16px"
      boxShadow="0px 4px 12px rgba(0, 0, 0, 0.1)"
    >
      {/* Header Section */}
      <Flex
        align="center"
        justify="center"
        direction="column"
        mb={6}
        position="relative"
      >
        <Flex align="center" justify="center" mb="12px">
          <SimmmpleEmojiIcon w="28px" h="28px" me="10px" mt="2px" />
          <Box
            bg="linear-gradient(97.89deg, #FFFFFF 70.67%, rgba(117, 122, 140, 0) 108.55%)"
            bgClip="text"
          >
            <Text
              fontSize="md"
              letterSpacing="3px"
              mt="3px"
              fontWeight="bold"
              color="transparent"
            >
              STEPS
            </Text>
          </Box>
        </Flex>
        <Divider orientation="horizontal" borderColor="gray.600" w="80%" />
      </Flex>

      {/* Timeline Steps */}
      <Stack spacing={6} position="relative" ml="16px">
        {steps.map((step, stepIndex) => {
          // major step 활성화 여부
          const isStepActive = stepIndex === currentStep;
          // subSteps 수에 따라 연결선 높이를 동적으로 계산 (예: subStep당 20px, 기본 오프셋 28px)
          const connectingLineHeight = `${step.subSteps.length * 20 + 28}px`;
          return (
            <Box key={stepIndex}>
              <Flex direction="row" align="flex-start">
                {/* 원형 아이콘 */}
                <Box position="relative">
                  <Box
                    w="16px"
                    h="16px"
                    borderRadius="full"
                    bg={isStepActive ? 'brand.400' : 'gray.500'}
                    border="2px solid white"
                  />
                  {/* 마지막 step이 아니라면 subSteps 수에 따라 동적으로 연결선 길이 적용 */}
                  {stepIndex < steps.length - 1 && (
                    <Box
                      position="absolute"
                      top="20px"
                      left="7px"
                      h={connectingLineHeight}
                      borderLeft="3px dashed rgba(255,255,255,0.3)"
                    />
                  )}
                </Box>
                {/* 텍스트 영역 */}
                <Box ml={6}>
                  <Text
                    fontWeight="bold"
                    color={isStepActive ? 'brand.400' : 'gray.400'}
                    fontSize="lg"
                  >
                    {step.title}
                  </Text>
                  <Stack mt={2} spacing={2}>
                    {step.subSteps.map((subStep, subIndex) => {
                      // 현재 substep 활성화 여부
                      const isSubActive =
                        isStepActive && currentSubStep === subIndex;
                      return (
                        <Link to={subStep.path} key={subIndex}>
                          <Text
                            fontSize="md"
                            color={isSubActive ? 'brand.100' : 'gray.300'}
                            _hover={{ color: 'blue.400' }}
                          >
                            {subStep.name}
                          </Text>
                        </Link>
                      );
                    })}
                  </Stack>
                </Box>
              </Flex>
            </Box>
          );
        })}
      </Stack>
    </Box>
  );
};

export default SidebarTimeline;
