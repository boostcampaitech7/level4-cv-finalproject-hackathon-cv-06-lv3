import React from 'react';
import { Flex, Text, Tooltip, Box } from '@chakra-ui/react';

const steps = [
  'Preprocessing Start',
  'Preprocessing Done',
  'Surrogate Training Start',
  'Surrogate Training Done',
  'Optimization Start',
  'Optimization Done',
];

export const FlowProgressIndicator = ({ progress }) => {
  return (
    <Flex align="center" justify="space-between" mt={2}>
      {steps.map((label, index) => {
        const stepNumber = index + 1;
        const isActive = stepNumber <= progress;
        return (
          <Tooltip key={index} label={label} hasArrow>
            <Box
              w="28px"
              h="28px"
              borderRadius="full"
              bg={isActive ? 'green.400' : 'gray.400'}
              display="flex"
              alignItems="center"
              justifyContent="center"
              fontSize="sm"
              color="white"
              fontWeight="bold"
            >
              {stepNumber}
            </Box>
          </Tooltip>
        );
      })}
    </Flex>
  );
};

export default FlowProgressIndicator;
