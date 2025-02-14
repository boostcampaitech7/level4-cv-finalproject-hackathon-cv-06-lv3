import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  Box,
  Flex,
  Icon,
} from '@chakra-ui/react';
import { CheckCircleIcon, ArrowRightIcon } from '@chakra-ui/icons';

const FlowBreadcrumb = ({ currentStep, onStepChange }) => {
  const steps = [
    { label: 'Dataset Upload', icon: <CheckCircleIcon /> },
    { label: 'Model Generation', icon: <ArrowRightIcon /> },
    { label: 'Goal Optimization', icon: <ArrowRightIcon /> },
    { label: 'Recommendation', icon: <ArrowRightIcon /> },
  ];

  return (
    <Flex
      direction="column"
      align="flex-start"
      p={4}
      w="200px"
      bg="gray.800"
      borderRadius="lg"
    >
      <Breadcrumb spacing="10px" separator="">
        {steps.map((step, index) => (
          <BreadcrumbItem key={index} isCurrentPage={currentStep === index}>
            <Flex
              align="center"
              cursor="pointer"
              onClick={() => onStepChange(index)}
            >
              <Box
                w="10px"
                h="10px"
                bg={currentStep === index ? 'blue.500' : 'gray.500'}
                borderRadius="full"
                mr={2}
              />
              <BreadcrumbLink fontSize="sm" fontWeight="bold" color="#fff">
                {step.icon} {step.label}
              </BreadcrumbLink>
            </Flex>
          </BreadcrumbItem>
        ))}
      </Breadcrumb>
    </Flex>
  );
};

export default FlowBreadcrumb;
