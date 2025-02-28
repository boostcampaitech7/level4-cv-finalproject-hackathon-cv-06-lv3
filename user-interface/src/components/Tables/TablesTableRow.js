import {
  Avatar,
  Badge,
  Flex,
  Td,
  Text,
  Tr,
  useColorModeValue,
} from '@chakra-ui/react';
import React from 'react';
import { FaGithub } from 'react-icons/fa';
import { SiNotion } from 'react-icons/si';

function TablesTableRow(props) {
  const {
    logo,
    name,
    email,
    subdomain,
    domain,
    status, // GitHub URL
    notion, // Notion URL
    lastItem,
  } = props;
  // 기본 텍스트 색상 등은 필요에 따라 설정
  const textColor = useColorModeValue('gray.700', 'white');

  return (
    <Tr>
      {/* Author Column */}
      <Td
        minWidth={{ sm: '250px' }}
        ps="0px"
        border={lastItem ? 'none' : null}
        borderBottomColor="#56577A"
      >
        <Flex align="center" py=".8rem" minWidth="100%" flexWrap="nowrap">
          <Avatar
            src={logo}
            w="50px"
            borderRadius="12px"
            me="18px"
            border="none"
          />
          <Flex direction="column">
            <Text
              fontSize="sm"
              color="#fff"
              fontWeight="normal"
              minWidth="100%"
            >
              {name}
            </Text>
            <Text fontSize="sm" color="gray.400" fontWeight="normal">
              {email}
            </Text>
          </Flex>
        </Flex>
      </Td>

      {/* Function/Details Column */}
      <Td
        border={lastItem ? 'none' : null}
        borderBottomColor="#56577A"
        minW="150px"
      >
        <Flex direction="column">
          <Text fontSize="sm" color="#fff" fontWeight="normal">
            {domain}
          </Text>
          <Text
            fontSize="sm"
            color="gray.400"
            fontWeight="normal"
            whiteSpace="pre-wrap"
          >
            {subdomain}
          </Text>
        </Flex>
      </Td>

      {/* Links Column: GitHub and Notion */}
      <Td border={lastItem ? 'none' : null} borderBottomColor="#56577A">
        <Flex align="center" gap={4}>
          <a href={status} target="_blank" rel="noopener noreferrer">
            <Flex align="center">
              <FaGithub color="blue.400" size="16px" />
              <Text fontSize="sm" color="blue.200" ml={1}>
                GitHub
              </Text>
            </Flex>
          </a>
          <a href={notion} target="_blank" rel="noopener noreferrer">
            <Flex align="center">
              <SiNotion color="white" size="16px" />
              <Text fontSize="sm" color="#fff" ml={1}>
                Notion
              </Text>
            </Flex>
          </a>
        </Flex>
      </Td>
    </Tr>
  );
}

export default TablesTableRow;
