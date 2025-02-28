import { Box, Flex, Text } from '@chakra-ui/react';

const FlowTimeline = ({ currentStep, onStepChange }) => {
  const steps = [
    'Dataset Upload',
    'Model Generation',
    'Goal Optimization',
    'Recommendation',
  ];

  return (
    <Flex
      direction="column"
      align="flex-start"
      bg="gray.800"
      p={4}
      borderRadius="lg"
      w="250px"
      h="calc(100vh - 20px)"
    >
      {steps.map((step, index) => (
        <Flex
          key={index}
          direction="column"
          align="center"
          w="100%"
          cursor="pointer"
          onClick={() => onStepChange(index)}
        >
          {/* 아이콘과 텍스트 */}
          <Box
            w="12px"
            h="12px"
            borderRadius="full"
            bg={currentStep === index ? 'blue.500' : 'gray.500'}
            mb={2}
          />
          <Text
            fontSize="sm"
            fontWeight={currentStep === index ? 'bold' : 'normal'}
            color={currentStep === index ? '#fff' : 'gray.400'}
            mb={4}
          >
            {step}
          </Text>
          {/* 단계 간의 선 */}
          {index < steps.length - 1 && (
            <Box
              w="2px"
              h="40px"
              bg={currentStep > index ? 'blue.500' : 'gray.500'}
              mb={4}
            />
          )}
        </Flex>
      ))}
    </Flex>
  );
};

export default FlowTimeline;
