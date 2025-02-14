import React from 'react';
import { Flex, Box, Tooltip } from '@chakra-ui/react';

// 각 단계에 대한 설명 (index 0부터 5까지)
const steps = [
  'Please move on to the next step!',
  'Preprocessing Done',
  'Surrogate Model Training Start',
  'Surrogate Model Training Done',
  'Optimization Started',
  'Optimization Done',
];

export const FlowProgressBar = ({ progress }) => {
  // progress: 1 ~ 6 (숫자)
  const totalSteps = steps.length;
  const segmentWidth = `${100 / totalSteps}%`;

  return (
    <Flex
      w="100%"
      h="12px"
      borderRadius="8px"
      overflow="hidden"
      bg="gray.300"
      mt={4}
    >
      {steps.map((label, index) => {
        const stepNumber = index + 1;
        // 완료된 단계는 green.400, 미완료는 회색
        const isCompleted = stepNumber <= progress;
        return isCompleted ? (
          <Tooltip key={index} label={label} hasArrow placement="top">
            <Box
              w={segmentWidth}
              h="100%"
              bg="green.400"
              transition="background-color 0.3s ease"
            />
          </Tooltip>
        ) : (
          <Box
            key={index}
            w={segmentWidth}
            h="100%"
            bg="gray.300"
            transition="background-color 0.3s ease"
          />
        );
      })}
    </Flex>
  );
};

export default FlowProgressBar;
