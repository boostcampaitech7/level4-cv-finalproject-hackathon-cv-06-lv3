import { Box, Button, Flex, Icon, Text, Tooltip } from '@chakra-ui/react';
import React from 'react';
import { FaPencilAlt, FaTrashAlt } from 'react-icons/fa';

function ProjectFileRow(props) {
  const {
    fileName,
    fileType,
    fileSize,
    isSelected,
    onSelect,
    onEdit,
    onDelete,
  } = props;

  return (
    <Box
      p="16px"
      bg={
        isSelected
          ? 'rgba(50, 50, 50, 0.8)'
          : 'linear-gradient(127.09deg, rgba(24, 29, 60, 0.94) 19.41%, rgba(10, 14, 35, 0.49) 76.65%)'
      }
      borderRadius="20px"
      cursor={isSelected ? 'not-allowed' : 'pointer'}
      transition="background-color 0.3s ease"
      _hover={{ bg: !isSelected ? 'rgba(6, 12, 41, 0.8)' : undefined }}
      onClick={!isSelected ? onSelect : undefined}
    >
      <Flex justify="space-between" alignItems="center" w="100%">
        {/* 파일 정보 */}
        <Flex direction="column" maxW="70%">
          <Tooltip
            label={fileName?.split('/').pop()}
            aria-label="File Name Tooltip"
          >
            <Text
              color="#fff"
              fontSize="sm"
              fontWeight="bold"
              mb="6px"
              isTruncated
              noOfLines={1}
              maxW="200px"
            >
              {fileName?.split('/').pop()}
            </Text>
          </Tooltip>
          <Text color="gray.400" fontSize="xs">
            File Type:{' '}
            <Text as="span" color="gray.500">
              {fileType}
            </Text>
          </Text>
          <Text color="gray.400" fontSize="xs">
            File Size:{' '}
            <Text as="span" color="gray.500">
              {fileSize}
            </Text>
          </Text>
        </Flex>

        {/* 액션 버튼 (수정 & 삭제) */}
        <Flex direction="row" alignItems="center" ml="auto" gap={0}>
          <Tooltip label={isSelected ? 'Cannot edit selected file' : 'Edit'}>
            <Button
              p="0px"
              variant="no-hover"
              isDisabled={isSelected} // ✅ 선택된 경우 버튼 비활성화
              cursor={isSelected ? 'not-allowed' : 'pointer'}
              onClick={e => {
                e.stopPropagation();
                if (!isSelected) onEdit();
              }}
            >
              <Flex
                color={isSelected ? 'gray.500' : 'green.500'}
                align="center"
                p="8px"
              >
                <Icon as={FaPencilAlt} me="2px" w="16px" h="16px" />
              </Flex>
            </Button>
          </Tooltip>

          <Tooltip
            label={isSelected ? 'Cannot delete selected file' : 'Delete'}
          >
            <Button
              p="0px"
              variant="no-hover"
              isDisabled={isSelected} // ✅ 선택된 경우 버튼 비활성화
              cursor={isSelected ? 'not-allowed' : 'pointer'}
              onClick={e => {
                e.stopPropagation();
                if (!isSelected) 
                  if (!window.confirm('Are you sure you want to delete this file?')) {
                    return;
                  }
                  onDelete();
              }}
            >
              <Flex
                color={isSelected ? 'gray.500' : 'red.500'}
                align="center"
                p="8px"
              >
                <Icon as={FaTrashAlt} me="2px" w="16px" h="16px" />
              </Flex>
            </Button>
          </Tooltip>
        </Flex>
      </Flex>
    </Box>
  );
}

export default ProjectFileRow;
