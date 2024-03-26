resource "aws_vpc" "vpc" {
  cidr_block = "192.168.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support = true

  tags = {
    "Name" = "mjh-vpc"
  }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.vpc.id
  tags = {
    "Name" = "mjh-igw"
  }
}

resource "aws_subnet" "subnet_a" {
  vpc_id = aws_vpc.vpc.id
  cidr_block = "192.168.0.0/24"
  availability_zone = "${var.region}a"
  enable_resource_name_dns_a_record_on_launch = true
  map_public_ip_on_launch = true
}

resource "aws_subnet" "subnet_c" {
  vpc_id = aws_vpc.vpc.id
  cidr_block = "192.168.1.0/24"
  availability_zone = "${var.region}c"
  enable_resource_name_dns_a_record_on_launch = true
  map_public_ip_on_launch = true
}

resource "aws_route_table" "public_route_table" {
  vpc_id = aws_vpc.vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = {
    "Name" = "mjh-public-route-table"
  }
}

resource "aws_route_table_association" "public_subnet_a_route_table_association" {
  subnet_id = aws_subnet.subnet_a.id
  route_table_id = aws_route_table.public_route_table.id
}

resource "aws_route_table_association" "public_subnet_c_route_table_association" {
  subnet_id = aws_subnet.subnet_c.id
  route_table_id = aws_route_table.public_route_table.id
}