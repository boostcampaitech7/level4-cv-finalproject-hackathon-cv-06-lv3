import { Box, Button, Flex, Grid, Icon, Text, Tooltip } from '@chakra-ui/react';
import React from 'react';
import { FaTrashAlt } from 'react-icons/fa';
import Card from './Card';

function SelectedFileCard({ fileName, onDeselect }) {
  return (
    <Card display="flex" alignItems="center" w="150px" size="xs" p={2}>
      <Flex direction="row" alignItems="center" justifyContent="space-between">
        <Tooltip label={fileName} aria-label="File Name Tooltip">
          <Text
            color="#fff"
            fontSize="xs"
            fontWeight="bold"
            isTruncated
            noOfLines={1}
            maxW="80px"
          >
            {fileName}
          </Text>
        </Tooltip>
        <Button p="0px" variant="no-hover" onClick={onDeselect}>
          <Flex color="red.500" cursor="pointer" align="center" p="8px">
            <Icon as={FaTrashAlt} w="12px" h="12px" />
          </Flex>
        </Button>
      </Flex>
    </Card>
  );
}

function SelectedDataArea({
  selectedFiles = [],
  onDeselect,
  allDatasets = [],
}) {
  // ✅ selectedFiles에서 csvId를 기준으로 allDatasets에서 파일명 찾기
  const selectedDatasets = selectedFiles.map(csvId => {
    const dataset = allDatasets.find(d => d.csvId === csvId);
    return dataset
      ? { csvId: dataset.csvId, fileName: dataset.csv.split('/').pop() }
      : { csvId, fileName: 'Unknown File' };
  });

  return (
    <Flex>
      {selectedDatasets.length > 0 ? (
        selectedDatasets.map(file => (
          <SelectedFileCard
            key={file.csvId}
            fileName={file.fileName} // ✅ 파일명만 표시
            onDeselect={() => onDeselect(file.csvId)} // ✅ csvId 넘겨서 삭제
          />
        ))
      ) : (
        <Text color="gray.400" textAlign="center" w="100%">
          No selected datasets.
        </Text>
      )}
    </Flex>
  );
}

export default SelectedDataArea;
